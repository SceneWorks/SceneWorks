//! rust-api catalog tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[test]
fn merge_model_manifest_entry_deep_merges_nested_blocks() {
    // The worker reads the merged manifest entry from the job payload now
    // (story 1653). This pins the behavior-preserving deep merge that the
    // worker's former `ltx_model_manifest_entry` performed: a user entry
    // overrides top-level keys, but builtin's siblings inside a nested block
    // (e.g. resources) survive rather than being replaced wholesale.
    let builtin = json!({
        "id": "ltx_2_3",
        "paths": {"model": "data/models/builtin"},
        "resources": {"checkpoint": {"path": "models/checkpoint.safetensors"}},
    });
    let user = json!({
        "id": "ltx_2_3",
        "paths": {"model": "data/models/user"},
        "resources": {"spatialUpscaler": {"path": "models/spatial.safetensors"}},
    });
    let merged = merge_model_manifest_entry(Some(builtin), Some(user));
    assert_eq!(merged["paths"]["model"], json!("data/models/user"));
    assert_eq!(
        merged["resources"]["checkpoint"]["path"],
        json!("models/checkpoint.safetensors")
    );
    assert_eq!(
        merged["resources"]["spatialUpscaler"]["path"],
        json!("models/spatial.safetensors")
    );
}

#[test]
fn merge_model_manifest_entry_handles_single_or_missing_sources() {
    let builtin = json!({"id": "ltx_2_3", "resources": {"checkpoint": {"path": "a"}}});
    assert_eq!(
        merge_model_manifest_entry(Some(builtin.clone()), None),
        builtin
    );
    let user = json!({"id": "ltx_2_3", "name": "user"});
    assert_eq!(merge_model_manifest_entry(None, Some(user.clone())), user);
    assert_eq!(merge_model_manifest_entry(None, None), json!({}));
}

#[test]
fn model_convert_request_parses_optional_mlx_quant_fields() {
    // The convert endpoint accepts optional camelCase quant knobs (sc-1982); the
    // worker reads the same field names off the job payload, so the contract must
    // hold. Absent fields default to None (unquantized bf16 conversion).
    let bare: crate::ModelConvertRequest = serde_json::from_value(json!({})).expect("bare body");
    assert_eq!(bare.quantize_bits, None);
    assert_eq!(bare.quantize_group_size, None);

    let quant: crate::ModelConvertRequest =
        serde_json::from_value(json!({"quantizeBits": 4, "quantizeGroupSize": 64}))
            .expect("quant body");
    assert_eq!(quant.quantize_bits, Some(4));
    assert_eq!(quant.quantize_group_size, Some(64));
}

#[test]
fn model_download_request_parses_optional_variant() {
    // sc-8508: the download endpoint accepts an optional quant `variant` (bf16/q8/q4) to install a
    // specific tier of a quant-matrix model. Absent = the default tier (back-compat single-variant).
    let bare: crate::ModelDownloadRequest = serde_json::from_value(json!({})).expect("bare body");
    assert_eq!(bare.variant, None);

    let tiered: crate::ModelDownloadRequest =
        serde_json::from_value(json!({ "variant": "q4" })).expect("variant body");
    assert_eq!(tiered.variant.as_deref(), Some("q4"));
}

#[test]
fn repo_slug_functions_match_cross_language_contract() {
    // story 1667: safe_download_dir is the api-only repo->dir slug op pinned by
    // the shared repo_slugs.json contract. (safe_repo_dir_name moved to
    // sceneworks-core in sc-4279 and is contract-tested there, so it is no longer
    // re-asserted here.)
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rust_migration_contracts/repo_slugs.json");
    let contract: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read repo_slugs.json"))
            .expect("parse repo_slugs.json");
    let cases = contract["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "repo_slugs fixture has no cases");
    for case in cases {
        let repo = case["repo"].as_str().expect("repo string");
        assert_eq!(
            safe_download_dir(repo),
            case["safeDownloadDir"].as_str().expect("safeDownloadDir"),
            "safe_download_dir drift for {repo:?}"
        );
    }
}

#[test]
fn mlx_catalog_status_reports_turnkey_and_conversion_states() {
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    // No `mlx` block -> None.
    let plain = json!({ "id": "z_image_turbo" });
    assert!(mlx_catalog_status(plain.as_object().unwrap(), &data_dir).is_none());

    // Turnkey model, repo not cached -> missing / ready.
    let ltx = json!({
        "id": "ltx_2_3",
        "mlx": { "minMemoryGb": 31, "repo": "SceneWorks/ltx-2.3-mlx" }
    });
    let status = mlx_catalog_status(ltx.as_object().unwrap(), &data_dir).expect("ltx status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "ready");

    // Turnkey model with the repo cached -> installed / ready.
    let repo_dir =
        huggingface_repo_cache_path(&data_dir, "SceneWorks/ltx-2.3-mlx").expect("repo cache path");
    std::fs::create_dir_all(repo_dir.join("snapshots")).expect("create snapshots");
    let status = mlx_catalog_status(ltx.as_object().unwrap(), &data_dir).expect("ltx status");
    assert_eq!(status.install_state, "installed");
    assert_eq!(status.conversion_state, "ready");

    // Conversion model, native source missing -> missing / needs_source.
    let wan5b = json!({
        "id": "wan_2_2",
        "mlx": {
            "minMemoryGb": 45,
            "requiresConversion": true,
            "convertSourceRepo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        }
    });
    let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "needs_source");

    // Native source cached -> missing / needs_conversion.
    let source_dir = huggingface_repo_cache_path(&data_dir, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        .expect("source cache path");
    std::fs::create_dir_all(source_dir.join("snapshots")).expect("create source snapshots");
    let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
    assert_eq!(status.conversion_state, "needs_conversion");

    // Converted MLX dir present -> installed / converted.
    let converted = data_dir.join("models").join("mlx").join("wan_2_2");
    std::fs::create_dir_all(&converted).expect("create converted dir");
    std::fs::write(converted.join("config.json"), "{}").expect("write config");
    let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
    assert_eq!(status.install_state, "installed");
    assert_eq!(status.conversion_state, "converted");
    assert_eq!(status.converted_path.unwrap(), converted);
}

#[test]
fn inject_converted_model_path_populates_modelpath_seam_once_converted() {
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    // A convert-at-install model (e.g. flux2_klein_9b_true_v2) whose local MLX dir
    // does not exist yet: leave `modelPath` unset so the worker reports the absent
    // conversion rather than silently loading the wrong source repo.
    let make_entry = || {
        json!({
            "id": "flux2_klein_9b_true_v2",
            "mlx": {
                "requiresConversion": true,
                "converter": "flux2_klein_diffusers",
                "convertSourceRepo": "wikeeyang/Flux2-Klein-9B-True-V2"
            }
        })
    };
    let mut entry = make_entry();
    inject_converted_model_path(&mut entry, &data_dir);
    assert!(
        entry.get("modelPath").is_none(),
        "modelPath must stay absent until the conversion has produced a local dir"
    );

    // Once the FLUX.2-klein converter has assembled the diffusers dir (marked by
    // model_index.json), `modelPath` is injected so the worker's resolve_weights_dir
    // loads the converted dir instead of falling back to the single-file source repo.
    let converted = data_dir
        .join("models")
        .join("mlx")
        .join("flux2_klein_9b_true_v2");
    std::fs::create_dir_all(&converted).expect("create converted dir");
    std::fs::write(converted.join("model_index.json"), "{}").expect("write model_index");
    let mut entry = make_entry();
    inject_converted_model_path(&mut entry, &data_dir);
    assert_eq!(
        entry.get("modelPath").and_then(Value::as_str),
        Some(converted.display().to_string().as_str()),
    );

    // An explicit manifest `modelPath` is authoritative and never overwritten.
    let mut pinned = make_entry();
    pinned
        .as_object_mut()
        .unwrap()
        .insert("modelPath".to_owned(), json!("/custom/path"));
    inject_converted_model_path(&mut pinned, &data_dir);
    assert_eq!(
        pinned.get("modelPath").and_then(Value::as_str),
        Some("/custom/path")
    );

    // A non-conversion model is untouched.
    let mut turnkey = json!({
        "id": "ltx_2_3",
        "mlx": { "repo": "SceneWorks/ltx-2.3-mlx" }
    });
    inject_converted_model_path(&mut turnkey, &data_dir);
    assert!(turnkey.get("modelPath").is_none());

    // FLUX.2-dev (sc-5921) converts to a packed Q4 dir whose top level is SUBDIRS
    // (transformer/ + text_encoder/, each with its own config.json) plus a symlinked
    // model_index.json — there is NO top-level config.json. The catalog's "converted"
    // detection keys on the top-level model_index.json, so the modelPath seam is still
    // injected for dev's subdir layout.
    let dev_entry = || {
        json!({
            "id": "flux2_dev",
            "mlx": {
                "requiresConversion": true,
                "converter": "flux2_dev_quant",
                "convertSourceRepo": "black-forest-labs/FLUX.2-dev"
            }
        })
    };
    let dev_converted = data_dir.join("models").join("mlx").join("flux2_dev");
    std::fs::create_dir_all(dev_converted.join("transformer")).expect("create dev transformer");
    std::fs::write(dev_converted.join("transformer").join("config.json"), "{}")
        .expect("write dev transformer config");
    // No top-level config.json — only the model_index.json marker.
    std::fs::write(dev_converted.join("model_index.json"), "{}").expect("write dev model_index");
    let mut entry = dev_entry();
    inject_converted_model_path(&mut entry, &data_dir);
    assert_eq!(
        entry.get("modelPath").and_then(Value::as_str),
        Some(dev_converted.display().to_string().as_str()),
        "dev's subdir layout is detected via its top-level model_index.json marker"
    );
}

#[tokio::test]
async fn real_builtin_catalog_exposes_krea_img2img_ui_flag() {
    // Regression (Michael, on-device): the "Image reference" img2img tile reads `selectedModel.ui.img2img`.
    // This runs the ACTUAL shipped repo manifest (not a fixture) through the real /api/v1/models catalog
    // path, proving the `ui.img2img` flag on krea_2_turbo survives merge → serialize to the response — so a
    // correct build DOES expose it (any on-device miss is a stale binary/config, not a code defect).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    // The canonical repo builtin manifest — the same bytes `sceneworks-core` embeds via include_str!.
    let real_manifest = include_str!("../../../../config/manifests/builtin.models.jsonc");
    std::fs::write(config_dir.join("builtin.models.jsonc"), real_manifest)
        .expect("builtin models writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let krea = models
        .as_array()
        .expect("catalog is an array")
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some("krea_2_turbo"))
        .expect("krea_2_turbo present in the catalog");
    assert_eq!(
        krea["ui"]["img2img"],
        Value::Bool(true),
        "krea_2_turbo must expose ui.img2img in the /models response (the img2img tile's gate)"
    );
    // And SD3.5 (A4.1) + Z-Image (A4.5, sc-10193) + Boogu (A4.3, sc-10191) + Ideogram (A4.4, sc-10192) —
    // the img2img flags added since — must be exposed the same way (a duplicate-`ui`-key drop would
    // silently strip the flag while still parsing, sc-10198). Ideogram is a `structuredPrompt` model, so
    // its flag additionally drives the img2img tile INSIDE the JSON-caption builder surface.
    for id in [
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "z_image",
        "z_image_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "ideogram_4",
        "ideogram_4_turbo",
        "sana_1600m",
        "sana_sprint_1600m",
    ] {
        let m = models
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m.get("id").and_then(Value::as_str) == Some(id))
            .unwrap_or_else(|| panic!("{id} present"));
        assert_eq!(
            m["ui"]["img2img"],
            Value::Bool(true),
            "{id} ui.img2img exposed"
        );
    }
}

#[tokio::test]
async fn models_catalog_carries_mac_support_and_capabilities_endpoint() {
    // sc-3486: the catalog stamps per-model `macSupport`, and the capabilities endpoint
    // carries the master switch (`macGatingActive` = mlx_required) + infra gating.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            { "id": "z_image_turbo", "name": "Z-Image-Turbo", "family": "z-image", "type": "image",
              "adapter": "z_image_diffusers", "capabilities": ["text_to_image"], "downloads": [],
              "paths": {}, "defaults": {}, "limits": {}, "loraCompatibility": { "families": [], "types": [] }, "ui": {} },
            { "id": "unported_image_model", "name": "Unported", "family": "unported", "type": "image",
              "adapter": "procedural_preview", "capabilities": ["text_to_image"], "downloads": [],
              "paths": {}, "defaults": {}, "limits": {}, "loraCompatibility": { "families": [], "types": [] }, "ui": {} },
            { "id": "svd", "name": "SVD", "family": "svd", "type": "video",
              "adapter": "svd_video", "capabilities": ["image_to_video"], "downloads": [],
              "paths": {}, "defaults": {}, "limits": {}, "loraCompatibility": { "families": [], "types": [] }, "ui": {} }
          ]
        }
        "#,
    )
    .expect("builtin models writes");

    let mut settings = test_settings(&temp_dir);
    settings.mlx_required = true;
    let app = create_app(settings).expect("app creates");

    let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let by_id = |id: &str| {
        models
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .cloned()
            .unwrap_or(Value::Null)
    };
    // Unported image model (no Rust/MLX engine) → unsupported on Mac, with a gap reason. No real
    // image model is torch-only anymore: every family was ported to MLX — Kolors (sc-3875),
    // PuLID-FLUX (sc-3344), and finally Lens / Lens-Turbo (epic 3164 / sc-5105, the last one) — so
    // the torch-only gating is demonstrated with a synthetic unported id, which has no dedicated
    // port epic (suggestedEpic absent → "needs an epic", epic 3482 policy).
    let torch_only = by_id("unported_image_model");
    assert_eq!(torch_only["macSupport"]["supported"], false);
    assert!(torch_only["macSupport"]["reason"].is_object());
    assert!(torch_only["macSupport"]["reason"]["suggestedEpic"].is_null());
    // MLX-routed family → supported, stays in the picker.
    assert_eq!(by_id("z_image_turbo")["macSupport"]["supported"], true);
    // SVD is now MLX-routed (sc-3523: `svd`→`svd_xt`, image→video only) → supported.
    assert_eq!(by_id("svd")["macSupport"]["supported"], true);

    // Capabilities endpoint: gating active (mlx_required=true) + infra epics present.
    let (status, caps) = request(app, "GET", "/api/v1/capabilities/mac", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(caps["macGatingActive"], true);
    assert_eq!(caps["notAvailableLabel"], "Not available on Mac (MLX only)");
    // Real-ESRGAN upscaling is ported to the Rust worker (sc-3489) → tool supported, no reason.
    assert_eq!(caps["features"]["imageUpscale"]["supported"], true);
    assert_eq!(caps["features"]["imageUpscale"]["reason"], Value::Null);
    // The AuraSR engine is dropped on Mac (sc-3668) AND off-Mac as an offered engine (sc-5499) → its
    // per-engine feature is unsupported on every platform and names the drop.
    assert_eq!(caps["features"]["imageUpscaleAuraSr"]["supported"], false);
    assert_eq!(
        caps["features"]["imageUpscaleAuraSr"]["reason"]["suggestedEpic"],
        "sc-5499"
    );
    // DWPose pose detection is ported to the Rust worker (sc-3487) → supported (sc-4206).
    assert_eq!(caps["features"]["poseFromPhoto"]["supported"], true);
    assert_eq!(caps["features"]["poseFromPhoto"]["reason"], Value::Null);
    // SeedVR2 video upscaling is net-new on Mac (epic 4811 / sc-4816) → supported.
    assert_eq!(caps["features"]["videoUpscale"]["supported"], true);
    assert_eq!(caps["features"]["videoUpscale"]["reason"], Value::Null);
    assert!(caps["training"]["supportedKernels"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k == "z_image_lora"));
    // Kolors training cut over to the native Rust trainer (sc-4732).
    assert!(caps["training"]["supportedKernels"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k == "kolors_lora"));
}

#[tokio::test]
async fn capabilities_mac_is_inert_when_mlx_not_required() {
    // The default (observe mode / Windows / Linux) reports gating inactive, so the client
    // applies no gating at all.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, caps) = request(app, "GET", "/api/v1/capabilities/mac", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(caps["macGatingActive"], false);
}

#[tokio::test]
async fn lora_download_endpoint_queues_hf_download_for_builtin_lora() {
    // sc-5944: built-in LoRAs gain an explicit Download (mirrors model download). A
    // catalog LoRA with a Hugging Face source queues a `lora_download` job carrying the
    // repo/file the worker fetches into the HF cache; a non-HF source or already-installed
    // LoRA is rejected, and an unknown id is 404.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "ltx_ic_union",
                  "name": "LTX IC Union",
                  "family": "ltx-video",
                  "compatibility": { "families": ["ltx-video"] },
                  "source": {
                    "provider": "huggingface",
                    "repo": "Lightricks/LTX-2.3-IC",
                    "file": "ic-union.safetensors"
                  }
                },
                {
                  "id": "local_only",
                  "name": "Local Only",
                  "family": "z-image",
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/local.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("builtin loras writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/loras/ltx_ic_union/download",
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_download");
    assert_eq!(job["payload"]["loraId"], "ltx_ic_union");
    assert_eq!(job["payload"]["loraName"], "LTX IC Union");
    assert_eq!(job["payload"]["provider"], "huggingface");
    assert_eq!(job["payload"]["repo"], "Lightricks/LTX-2.3-IC");
    assert_eq!(job["payload"]["files"][0], "ic-union.safetensors");
    assert_eq!(job["payload"]["family"], "ltx-video");

    // A LoRA whose source is not a Hugging Face repo can't be fetched this way.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/loras/local_only/download",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // An unknown LoRA id is a 404.
    let (status, _) = request(app, "POST", "/api/v1/loras/missing/download", json!({})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn model_and_lora_routes_match_manifest_behavior() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "base-model",
                  "name": "Base Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [
                    { "provider": "huggingface", "repo": "owner/alternate-model", "files": ["*.bin"], "estimatedSizeBytes": 536870912 },
                    { "provider": "huggingface", "repo": "owner/model", "files": ["*.safetensors"], "default": true, "estimatedSizeBytes": 12884901888 }
                  ],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": { "label": "Base" }
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "base-model",
                  "name": "User Model",
                  "ui": { "label": "User" },
                  "customPluginMetadata": { "vendorKey": "preserved" }
                }
              ]
            }
            "#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style-lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": ["style"],
                  "compatibility": { "families": ["z-image", "wan-video"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "cinematic",
                  "name": "Cinematic",
                  "workflow": "text_to_image",
                  "model": "base-model",
                  "defaults": { "count": 4, "resolution": "1280x720", "negativePrompt": "flat lighting" },
                  "prompt": { "suffix": "cinematic lighting" },
                  "loras": [{ "id": "style-lora", "weight": 0.5 }]
                }
              ]
            }
            "#,
        )
        .expect("builtin recipe presets writes");
    std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                { "id": "cinematic", "name": "My Cinematic", "defaults": { "count": 2, "resolution": "1280x720", "negativePrompt": "flat lighting" } },
                { "id": "legacy_edit", "name": "Legacy Edit", "modes": ["edit_image"], "builtInLoras": [{ "id": "style-lora", "weight": 0.25 }] }
              ]
            }
            "#,
        )
        .expect("user recipe presets writes");
    let marker_dir = temp_dir.path().join("data/models/owner__model");
    std::fs::create_dir_all(&marker_dir).expect("model dir creates");
    std::fs::write(marker_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("marker writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["name"], "User Model");
    // sc-12338: schema enforcement is builtin-only. User manifests remain a lenient
    // extension surface; an unknown custom key must neither brick the route nor be dropped.
    assert_eq!(models[0]["customPluginMetadata"]["vendorKey"], "preserved");
    assert_eq!(models[0]["adapter"], "z_image_diffusers");
    assert_eq!(models[0]["downloadable"], true);
    assert_eq!(models[0]["downloadSizeBytes"], 12884901888_u64);
    assert_eq!(models[0]["downloadSizeLabel"], "12.0 GB");
    assert_eq!(models[0]["downloadSizeEstimated"], true);
    assert_eq!(models[0]["installState"], "installed");
    assert!(models[0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.ends_with("owner__model")));

    let (status, loras) = request(
        app.clone(),
        "GET",
        "/api/v1/loras?modelFamily=wan-video",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(loras.as_array().unwrap().len(), 1);
    assert_eq!(loras[0]["id"], "style-lora");

    let (status, presets) =
        request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let presets = presets.as_array().unwrap();
    assert_eq!(presets.len(), 2);
    let cinematic = presets
        .iter()
        .find(|preset| preset["id"] == "cinematic")
        .expect("cinematic preset");
    assert_eq!(cinematic["name"], "My Cinematic");
    assert_eq!(cinematic["scope"], "global");
    assert_eq!(cinematic["workflow"], "text_to_image");
    assert_eq!(cinematic["defaults"]["count"], 2);
    assert_eq!(cinematic["loras"][0]["id"], "style-lora");
    assert_eq!(cinematic["builtInLoras"][0]["id"], "style-lora");
    let legacy_edit = presets
        .iter()
        .find(|preset| preset["id"] == "legacy_edit")
        .expect("legacy edit preset");
    assert_eq!(legacy_edit["workflow"], "edit_image");
    assert_eq!(legacy_edit["model"], "base-model");
    assert_eq!(legacy_edit["loras"][0]["id"], "style-lora");
    assert_eq!(
        legacy_edit["appliedDefaults"]["notes"][0],
        "workflow inferred from legacy modes as edit_image"
    );
    assert_eq!(
        legacy_edit["appliedDefaults"]["notes"][1],
        "model defaulted to base-model for legacy preset"
    );
    assert_eq!(
        legacy_edit["appliedDefaults"]["notes"][2],
        "builtInLoras migrated to loras"
    );

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "city at night",
            "model": "base-model",
            // Client render settings that DIFFER from the preset's declared
            // defaults (count 2 / 1280x720 / "flat lighting") — the studio seeds
            // the form from the preset but the user can override, so these
            // submitted values must win.
            "count": 1,
            "width": 512,
            "height": 512,
            "negativePrompt": "client negative prompt",
            "recipePresetId": "cinematic",
            "advanced": { "resolution": "512x512" }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        image_job["payload"]["prompt"],
        "city at night, cinematic lighting"
    );
    assert_eq!(image_job["payload"]["loras"][0]["id"], "style-lora");
    assert!(image_job["payload"]["loras"][0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.ends_with("data/loras/style.safetensors")
            || value.ends_with("data\\loras\\style.safetensors")
            || value.ends_with("loras/style.safetensors")
            || value.ends_with("loras\\style.safetensors")));
    assert_eq!(
        image_job["payload"]["loras"][0]["source"]["path"],
        "loras/style.safetensors"
    );
    assert_eq!(image_job["payload"]["model"], "base-model");
    assert_eq!(image_job["payload"]["loras"][0]["family"], "z-image");
    assert_eq!(
        image_job["payload"]["loras"][0]["compatibility"]["families"][0],
        "z-image"
    );
    // Render settings are client-owned and overrideable: the submitted values
    // win over the preset's declared defaults.
    assert_eq!(image_job["payload"]["count"], 1);
    assert_eq!(image_job["payload"]["seeds"].as_array().unwrap().len(), 1);
    assert_eq!(image_job["payload"]["width"], 512);
    assert_eq!(image_job["payload"]["height"], 512);
    assert_eq!(
        image_job["payload"]["negativePrompt"],
        "client negative prompt"
    );
    assert_eq!(image_job["payload"]["advanced"]["resolution"], "512x512");
    assert_eq!(
        image_job["payload"]["advanced"]["recipePresetId"],
        "cinematic"
    );

    let (status, null_path_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "city with selected lora",
            "model": "base-model",
            "count": 1,
            "width": 512,
            "height": 512,
            "loras": [{
                "id": "style-lora",
                "name": null,
                "triggerWords": null,
                "compatibility": null,
                "installedPath": null,
                "sourcePath": null
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(null_path_job["payload"]["loras"][0]["id"], "style-lora");
    assert_eq!(null_path_job["payload"]["loras"][0]["name"], "Style LoRA");
    assert_eq!(
        null_path_job["payload"]["loras"][0]["triggerWords"][0],
        "style"
    );
    assert_eq!(
        null_path_job["payload"]["loras"][0]["compatibility"]["families"][0],
        "z-image"
    );
    assert!(null_path_job["payload"]["loras"][0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.ends_with("data/loras/style.safetensors")
            || value.ends_with("data\\loras\\style.safetensors")
            || value.ends_with("loras/style.safetensors")
            || value.ends_with("loras\\style.safetensors")));

    let (status, preset_model_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "city at dawn",
            "count": 1,
            "width": 512,
            "height": 512,
            "recipePresetId": "cinematic"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(preset_model_job["payload"]["model"], "base-model");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/models/base-model/download",
        json!({ "requestedGpu": "" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "model_download");
    assert_eq!(job["requestedGpu"], "auto");
    assert_eq!(job["payload"]["modelName"], "User Model");
    assert_eq!(job["payload"]["repo"], "owner/model");
    assert_eq!(job["payload"]["files"][0], "*.safetensors");
    assert_eq!(job["payload"]["targetDir"], models[0]["installedPath"]);

    let (status, job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({ "repo": "owner/lora", "name": "Imported LoRA", "files": ["adapter.safetensors"] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_import");
    assert_eq!(job["payload"]["repo"], "owner/lora");
    assert_eq!(job["payload"]["loraId"], "imported_lora");
    assert_eq!(job["payload"]["scope"], "global");
    assert!(job["payload"]["targetDir"]
        .as_str()
        .is_some_and(|value| value.ends_with("data/loras/imported_lora")
            || value.ends_with("data\\loras\\imported_lora")));
    assert_eq!(job["payload"]["manifestEntry"]["scope"], "global");
    assert!(job["payload"].get("sourcePath").is_none());

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, url_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourceUrl": "https://example.com/loras/detail.safetensors",
            "name": "Detail LoRA",
            "family": "z-image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(url_job["type"], "lora_import");
    assert_eq!(
        url_job["payload"]["sourceUrl"],
        "https://example.com/loras/detail.safetensors"
    );
    // sc-10214: a declared family scopes the id/folder (`z-image` → `z_image_` prefix).
    assert_eq!(url_job["payload"]["loraId"], "z_image_detail_lora");
    assert_eq!(
        url_job["payload"]["manifestEntry"]["source"]["provider"],
        "url"
    );
    assert_eq!(
        url_job["payload"]["manifestEntry"]["source"]["url"],
        "https://example.com/loras/detail.safetensors"
    );
    assert_eq!(url_job["payload"]["manifestEntry"]["family"], "z-image");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let upload_bytes = test_safetensors_bytes();
    let (status, upload_job) = request_multipart_lora_upload(
        app,
        &[
            ("name", "Uploaded Detail"),
            ("scope", "global"),
            ("family", "z-image"),
        ],
        "detail.safetensors",
        &upload_bytes,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(upload_job["type"], "lora_import");
    assert_eq!(upload_job["payload"]["loraId"], "z_image_uploaded_detail");
    assert_eq!(upload_job["payload"]["uploadedSourcePath"], true);
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["source"]["provider"],
        "local"
    );
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["files"][0],
        "detail.safetensors"
    );
    let source_path = std::path::PathBuf::from(
        upload_job["payload"]["sourcePath"]
            .as_str()
            .expect("source path"),
    );
    assert_eq!(
        std::fs::read(&source_path).expect("staged upload reads"),
        upload_bytes
    );
    assert_eq!(
        source_path.file_name().and_then(|value| value.to_str()),
        Some("detail.safetensors")
    );

    TEST_MAX_LORA_UPLOAD_BYTES.with(|cap| cap.set(4));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (bad_status, bad_error) = request_multipart_lora_upload(
        app,
        &[("name", "Too Large"), ("scope", "global")],
        "too-large.safetensors",
        b"12345",
    )
    .await;
    TEST_MAX_LORA_UPLOAD_BYTES.with(|cap| cap.set(0));
    assert_eq!(bad_status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        bad_error["detail"],
        "Uploaded LoRA file exceeds the 2GB limit"
    );

    let lora_source_dir = temp_dir.path().join("data").join("loras");
    std::fs::create_dir_all(&lora_source_dir).expect("lora source dir creates");
    let lora_source = lora_source_dir.join("safe-local.safetensors");
    write_test_safetensors(&lora_source);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, source_path_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": lora_source.display().to_string(),
            "name": "Safe Local Source"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        source_path_job["payload"]["manifestEntry"]["source"]["provider"],
        "local"
    );

    let outside_source = temp_dir.path().join("outside.safetensors");
    write_test_safetensors(&outside_source);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (bad_status, bad_error) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": outside_source.display().to_string(),
            "name": "Unsafe Local Source"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
            bad_error["detail"],
            "LoRA sourcePath must be inside app-managed data/loras, project/loras, or staged upload folders"
        );

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/loras/import",
        json!({ "sourceUrl": "file:///tmp/detail.safetensors" }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(bad_error["detail"], "LoRA sourceUrl must use http or https");

    let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/loras/import",
            json!({ "sourceUrl": "https://example.com/loras/detail.safetensors", "family": "unknown-family" }),
        )
        .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Unsupported LoRA family: unknown-family"
    );

    let (status, normalized_family) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourceUrl": "https://example.com/loras/z-detail.safetensors",
            "family": "Z_Image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        normalized_family["payload"]["manifestEntry"]["family"],
        "z-image"
    );

    // sc-1378: architecture detection at import time. The detection
    // policy lets the user supply any family the catalog declares, so
    // expand the catalog now to include the families we exercise below.
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "base-model",
                  "name": "Base Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [
                    { "provider": "huggingface", "repo": "owner/alternate-model", "files": ["*.bin"], "estimatedSizeBytes": 536870912 },
                    { "provider": "huggingface", "repo": "owner/model", "files": ["*.safetensors"], "default": true, "estimatedSizeBytes": 12884901888 }
                  ],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": { "label": "Base" }
                },
                {
                  "id": "qwen-image-base",
                  "name": "Qwen Image",
                  "family": "qwen-image",
                  "type": "image",
                  "adapter": "qwen_image",
                  "capabilities": ["text_to_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                },
                {
                  "id": "wan-video-base",
                  "name": "Wan Video",
                  "family": "wan-video",
                  "type": "video",
                  "adapter": "wan_video",
                  "capabilities": ["text_to_video"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models rewrites for detection tests");

    let detect_dir = temp_dir.path().join("data").join("loras");
    std::fs::create_dir_all(&detect_dir).expect("detect dir creates");

    // Qwen-Image-shaped file with a mismatched user-supplied family is
    // rejected with both values surfaced in the error message.
    let mismatch_path = detect_dir.join("qwen-as-wan.safetensors");
    write_test_safetensors_with_keys(&mismatch_path, &qwen_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, mismatch_error) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": mismatch_path.display().to_string(),
            "family": "wan-video",
            "name": "Mismatched"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let detail = mismatch_error["detail"].as_str().expect("detail string");
    assert!(
        detail.contains("qwen-image") && detail.contains("wan-video"),
        "mismatch error must surface both detected and supplied families, got: {detail}"
    );

    // Low-block MMDiT tensors are inconclusive rather than treated as
    // Z-Image; sparse Qwen LoRAs can target only early blocks.
    let auto_path = detect_dir.join("low-mmdit-no-autofill.safetensors");
    write_test_safetensors_with_keys(&auto_path, &z_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, auto_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": auto_path.display().to_string(),
            "name": "Auto Family"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(auto_job["payload"]["manifestEntry"].get("family").is_none());

    // Supplied family + inconclusive MMDiT detection succeeds, and the
    // user-supplied family is kept.
    let match_path = detect_dir.join("z-match.safetensors");
    write_test_safetensors_with_keys(&match_path, &z_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, match_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": match_path.display().to_string(),
            "family": "z-image",
            "name": "Matched"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(match_job["payload"]["manifestEntry"]["family"], "z-image");

    // Wan-shaped tensors are detected and accepted when the user agrees.
    let wan_match_path = detect_dir.join("wan-match.safetensors");
    write_test_safetensors_with_keys(&wan_match_path, &wan_video_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, wan_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": wan_match_path.display().to_string(),
            "family": "wan-video",
            "name": "Wan Matched"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(wan_job["payload"]["manifestEntry"]["family"], "wan-video");

    // Inconclusive header (only `__metadata__`) + supplied family is
    // accepted unchanged — the user-supplied label survives.
    let inconclusive_path = detect_dir.join("inconclusive.safetensors");
    write_test_safetensors(&inconclusive_path);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, inconclusive_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": inconclusive_path.display().to_string(),
            "family": "z-image",
            "name": "Inconclusive"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        inconclusive_job["payload"]["manifestEntry"]["family"],
        "z-image"
    );

    // Confident Qwen-Image detection (block count > 40) auto-fills.
    let qwen_path = detect_dir.join("qwen-autofill.safetensors");
    write_test_safetensors_with_keys(&qwen_path, &qwen_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, qwen_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": qwen_path.display().to_string(),
            "name": "Qwen Auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(qwen_job["payload"]["manifestEntry"]["family"], "qwen-image");
}

#[tokio::test]
async fn paired_moe_lora_upload_writes_convention_files_and_records_base_model() {
    // sc-1991: a bring-your-own Wan A14B MoE pair uploads as two file parts
    // (`file` = high-noise, `secondaryFile` = low-noise) under one record. The
    // import normalizes both halves to the dot-delimited high/low_noise convention
    // (off-convention upload names included) and persists the chosen A14B baseModel.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let wan_bytes = test_safetensors_bytes_with_keys(&wan_video_tensor_keys());
    let (status, job) = request_multipart_lora_pair_upload(
        app,
        &[
            ("name", "Wan MoE"),
            ("scope", "global"),
            ("family", "wan-video"),
            ("baseModel", "wan_2_2_t2v_14b"),
        ],
        // Community names that do NOT match the convention — must be normalized.
        ("high_noise_model.safetensors", &wan_bytes),
        ("low_noise_model.safetensors", &wan_bytes),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_import");
    let lora_id = job["payload"]["loraId"].as_str().expect("loraId");
    assert_eq!(job["payload"]["manifestEntry"]["family"], "wan-video");
    assert_eq!(
        job["payload"]["manifestEntry"]["baseModel"],
        "wan_2_2_t2v_14b"
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["files"][0],
        format!("{lora_id}.high_noise.safetensors")
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["files"][1],
        format!("{lora_id}.low_noise.safetensors")
    );

    // Both halves staged on disk; the worker renames them on import.
    let primary = std::path::PathBuf::from(
        job["payload"]["sourcePath"]
            .as_str()
            .expect("primary source path"),
    );
    let secondary = std::path::PathBuf::from(
        job["payload"]["secondarySourcePath"]
            .as_str()
            .expect("secondary source path"),
    );
    assert_eq!(std::fs::read(&primary).expect("primary reads"), wan_bytes);
    assert_eq!(
        std::fs::read(&secondary).expect("secondary reads"),
        wan_bytes
    );
    assert_ne!(primary, secondary);
}

#[tokio::test]
async fn model_import_route_is_disabled_on_every_platform() {
    // sc-7081 (epic 7080): model upload/import is intentionally disabled until a real
    // compatibility + conversion pipeline exists. Both the JSON and multipart entrypoints
    // must short-circuit with an actionable 403 before any staging/queueing. The deeper
    // validation/family-detection logic still lives behind `create_model_import_job` (and is
    // covered by lora_family + worker tests); restore the route-level assertions when the
    // feature is re-enabled.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    for (name, body) in [
        (
            "builtin.models.jsonc",
            r#"{ "schemaVersion": 1, "models": [] }"#,
        ),
        (
            "user.models.jsonc",
            r#"{ "schemaVersion": 1, "models": [] }"#,
        ),
        (
            "builtin.loras.jsonc",
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        ),
        ("user.loras.jsonc", r#"{ "schemaVersion": 1, "loras": [] }"#),
        (
            "builtin.recipe-presets.jsonc",
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        ),
        (
            "user.recipe-presets.jsonc",
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        ),
    ] {
        std::fs::write(config_dir.join(name), body).expect("manifest writes");
    }

    // JSON path: refused before any validation or queueing.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/models/import",
        json!({
            "sourceUrl": "https://example.com/models/custom.safetensors",
            "type": "image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(error["detail"].as_str().unwrap_or("").contains("disabled"));

    // Multipart path: refused before the upload is staged.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (mp_status, mp_error) = request_multipart_model_upload(
        app,
        &[("name", "Disabled"), ("type", "image")],
        "disabled.safetensors",
        &test_safetensors_bytes_with_keys(&qwen_image_tensor_keys()),
    )
    .await;
    assert_eq!(mp_status, StatusCode::FORBIDDEN);
    assert!(mp_error["detail"]
        .as_str()
        .unwrap_or("")
        .contains("disabled"));
}

#[tokio::test]
async fn imported_model_catalog_uses_paths_model_install_marker() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let model_dir = temp_dir.path().join("data/models/imports/custom_model");
    std::fs::create_dir_all(&model_dir).expect("model dir creates");
    std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("marker writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        format!(
            r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "custom_model",
                    "name": "Custom Model",
                    "type": "image",
                    "family": "z-image",
                    "paths": {{ "model": "{}" }}
                  }}]
                }}"#,
            model_dir.display().to_string().replace('\\', "\\\\")
        ),
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "custom_model");
    assert_eq!(models[0]["downloadable"], false);
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["installedPath"], model_dir.display().to_string());
}

#[tokio::test]
async fn downloadable_model_catalog_uses_huggingface_cache_install_state() {
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "base_model",
                "name": "Base Model",
                "type": "image",
                "family": "z-image",
                "downloads": [{ "provider": "huggingface", "repo": "owner/model" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--model/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "StableDiffusionPipeline",
          "scheduler": ["diffusers", "DDPMScheduler"],
          "unet": ["diffusers", "UNet2DConditionModel"],
          "vae": ["diffusers", "AutoencoderKL"],
          "tokenizer": ["transformers", "CLIPTokenizer"]
        }"#,
    )
    .expect("model index writes");
    for (dir, file) in [
        ("scheduler", "scheduler_config.json"),
        ("unet", "config.json"),
        ("vae", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    std::fs::write(
        cache_dir.join("unet/diffusion_pytorch_model.safetensors"),
        "weights",
    )
    .expect("unet weights write");
    std::fs::write(
        cache_dir.join("vae/diffusion_pytorch_model.safetensors"),
        "weights",
    )
    .expect("vae weights write");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "base_model");
    assert_eq!(models[0]["downloadable"], true);
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert!(models[0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.contains("models--owner--model")));
}

#[tokio::test]
async fn downloadable_model_catalog_flags_incomplete_huggingface_snapshots() {
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "sdxl",
                "name": "SDXL",
                "type": "image",
                "family": "sdxl",
                "downloads": [{ "provider": "huggingface", "repo": "owner/sdxl" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--sdxl/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "StableDiffusionXLPipeline",
          "scheduler": ["diffusers", "EulerDiscreteScheduler"],
          "unet": ["diffusers", "UNet2DConditionModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    for (dir, file) in [
        ("scheduler", "scheduler_config.json"),
        ("unet", "config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    std::fs::write(
        cache_dir.join("unet/diffusion_pytorch_model.safetensors"),
        "weights",
    )
    .expect("unet weights write");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "sdxl");
    assert_eq!(models[0]["installState"], "missing");
    assert_eq!(models[0]["cacheState"], "incomplete");
    assert_eq!(models[0]["repairAvailable"], true);
    assert_eq!(
        models[0]["missingRequiredFiles"],
        json!(["vae/<weights>", "vae/config.json"])
    );
    assert!(models[0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.contains("models--owner--sdxl")));
}

#[tokio::test]
async fn downloadable_model_catalog_ignores_absent_optional_diffusers_components() {
    // Chroma's model_index.json declares `feature_extractor` and `image_encoder`
    // as `[null, null]` — diffusers' sentinel for optional components the pipeline
    // doesn't use, which have no files on disk by design. The health check must
    // not report them as missing, otherwise a fully-installed model is flagged
    // incomplete on every platform.
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "chroma1_base",
                "name": "Chroma1-Base",
                "type": "image",
                "family": "chroma",
                "downloads": [{ "provider": "huggingface", "repo": "lodestones/Chroma1-Base" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--lodestones--Chroma1-Base/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "ChromaPipeline",
          "feature_extractor": [null, null],
          "image_encoder": [null, null],
          "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
          "text_encoder": ["transformers", "T5EncoderModel"],
          "tokenizer": ["transformers", "T5Tokenizer"],
          "transformer": ["diffusers", "ChromaTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    for (dir, file) in [
        ("scheduler", "scheduler_config.json"),
        ("text_encoder", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("transformer", "config.json"),
        ("vae", "config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    for dir in ["text_encoder", "transformer", "vae"] {
        std::fs::write(
            cache_dir
                .join(dir)
                .join("diffusion_pytorch_model.safetensors"),
            "weights",
        )
        .expect("component weights write");
    }

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "chroma1_base");
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert_eq!(models[0]["missingRequiredFiles"], json!([]));
}

#[tokio::test]
async fn downloadable_model_catalog_treats_processor_components_as_weightless() {
    // Qwen-Image-Edit-2511's model_index.json declares a `processor`
    // (Qwen2VLProcessor) — a transformers preprocessing wrapper that carries no
    // model weights and ships `preprocessor_config.json` instead of `config.json`.
    // The health check must not demand `processor/config.json` + `processor/<weights>`
    // (which the repo never contains), otherwise a fully-installed model is flagged
    // incomplete forever and the "Fix" download can never satisfy it.
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "qwen_image_edit_2511",
                "name": "Qwen-Image-Edit-2511",
                "type": "image",
                "family": "qwen-image",
                "downloads": [{ "provider": "huggingface", "repo": "Qwen/Qwen-Image-Edit-2511" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "QwenImageEditPlusPipeline",
          "processor": ["transformers", "Qwen2VLProcessor"],
          "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
          "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
          "tokenizer": ["transformers", "Qwen2Tokenizer"],
          "transformer": ["diffusers", "QwenImageTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKLQwenImage"]
        }"#,
    )
    .expect("model index writes");
    // processor ships preprocessor_config.json + tokenizer files, but no
    // config.json and no weights — exactly the real Qwen2VLProcessor layout.
    for (dir, file) in [
        ("processor", "preprocessor_config.json"),
        ("scheduler", "scheduler_config.json"),
        ("text_encoder", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("transformer", "config.json"),
        ("vae", "config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    for dir in ["text_encoder", "transformer", "vae"] {
        std::fs::write(
            cache_dir
                .join(dir)
                .join("diffusion_pytorch_model.safetensors"),
            "weights",
        )
        .expect("component weights write");
    }

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "qwen_image_edit_2511");
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert_eq!(models[0]["missingRequiredFiles"], json!([]));
}

#[tokio::test]
async fn downloadable_model_catalog_reports_incomplete_cache_for_installed_managed_model() {
    // A model can have SceneWorks' managed completion marker while its Hugging
    // Face cache snapshot is partial. Keep those states independent so the UI
    // can offer "Fix" instead of disabling the primary action as "Ready".
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let repo = "owner/mixed";
    single_model_manifest(
        &temp_dir.path().join("config/manifests"),
        "mixed_model",
        repo,
    );
    let managed_dir = temp_dir
        .path()
        .join("data/models")
        .join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed_dir).expect("managed dir creates");
    std::fs::write(managed_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("managed marker writes");
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--mixed/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("partial hf cache creates");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "incomplete");
    assert_eq!(models[0]["repairAvailable"], true);
    assert_eq!(
        models[0]["missingRequiredFiles"],
        json!(["model_index.json"])
    );
}

#[test]
fn huggingface_cache_health_accepts_readable_snapshot_symlinked_model_index() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let repo_root = temp_dir.path().join("models--owner--symlinked");
    let snapshot = repo_root.join("snapshots/abc123");
    let blobs = repo_root.join("blobs");
    std::fs::create_dir_all(&snapshot).expect("snapshot creates");
    std::fs::create_dir_all(&blobs).expect("blobs creates");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs creates");
    std::fs::write(repo_root.join("refs/main"), "abc123").expect("ref writes");
    let blob_name = "model-index-blob";
    std::fs::write(
        blobs.join(blob_name),
        r#"{ "_class_name": "EmptyPipeline" }"#,
    )
    .expect("blob writes");
    let link = snapshot.join("model_index.json");
    let relative_target = std::path::Path::new("..")
        .join("..")
        .join("blobs")
        .join(blob_name);
    if create_test_symlink_file(&relative_target, &link).is_err() {
        std::fs::write(&link, r#"{ "_class_name": "EmptyPipeline" }"#)
            .expect("fallback index writes");
    }

    let health = crate::models::huggingface_cache_health(&repo_root, &[]);

    assert!(health.installed);
    assert!(!health.incomplete);
    assert!(health.missing_files.is_empty());
}

#[test]
fn turbo_cache_health_flags_missing_lora_only_when_listed_explicitly() {
    // mlx-gen #488: a user who downloaded base Ideogram 4 BEFORE the TurboTime LoRA was published
    // shares this repo's snapshot — q4/ holds the base files but NOT turbo_lora.safetensors. A coarse
    // `q4/*` glob is already satisfied by the base files, so the missing LoRA never flags
    // "incomplete" (no Fix button — the reported symptom). Listing the exact path makes the check
    // verify that specific file.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let repo_root = temp_dir.path().join("models--SceneWorks--ideogram-4-mlx");
    let q4 = repo_root.join("snapshots/abc123/q4");
    std::fs::create_dir_all(&q4).expect("q4 creates");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs creates");
    std::fs::write(repo_root.join("refs/main"), "abc123").expect("ref writes");
    // A base file under q4/ so the `q4/*` glob is satisfied — but the LoRA is absent.
    std::fs::write(q4.join("model_index.json"), "{}").expect("base file writes");

    let glob_only = vec!["q4/*".to_owned()];
    let with_specific = vec!["q4/*".to_owned(), "q4/turbo_lora.safetensors".to_owned()];

    // The coarse glob alone misses it (the bug): q4/model_index.json satisfies `q4/*`.
    let coarse = crate::models::huggingface_cache_health(&repo_root, &glob_only);
    assert!(
        coarse.installed && !coarse.incomplete,
        "coarse `q4/*` glob is satisfied by base files and falsely reports complete"
    );

    // The explicit path catches the missing LoRA → incomplete → Fix button.
    let precise = crate::models::huggingface_cache_health(&repo_root, &with_specific);
    assert!(
        precise.incomplete,
        "explicit `q4/turbo_lora.safetensors` must flag the missing file"
    );
    assert!(
        precise
            .missing_files
            .iter()
            .any(|file| file == "q4/turbo_lora.safetensors"),
        "missing_files must name the LoRA, got {:?}",
        precise.missing_files
    );

    // Once the LoRA is present, the model is complete again.
    std::fs::write(q4.join("turbo_lora.safetensors"), "weights").expect("lora writes");
    let fixed = crate::models::huggingface_cache_health(&repo_root, &with_specific);
    assert!(
        fixed.installed && !fixed.incomplete && fixed.missing_files.is_empty(),
        "complete once the LoRA is present"
    );
}

#[test]
fn cache_health_does_not_let_an_appledouble_sidecar_satisfy_a_required_pattern() {
    // SceneWorks#1333: `._model.safetensors` is a macOS AppleDouble sidecar, not weights. It
    // carries the `.safetensors` extension, so a `*.safetensors` glob matched it and the model
    // reported INSTALLED while the real weights file was absent — the worst failure mode here,
    // because the load then dies at generation time with a corrupt-header error.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let repo_root = temp_dir.path().join("models--SceneWorks--boogu-image-mlx");
    let mllm = repo_root.join("snapshots/abc123/base/mllm");
    std::fs::create_dir_all(&mllm).expect("mllm creates");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs creates");
    std::fs::write(repo_root.join("refs/main"), "abc123").expect("ref writes");
    std::fs::write(mllm.join("config.json"), "{}").expect("config writes");
    // A real AppleDouble header (magic 0x00051607, version 0x00020000) — and NO real weights.
    std::fs::write(
        mllm.join("._model.safetensors"),
        [0x00, 0x05, 0x16, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00],
    )
    .expect("sidecar writes");

    let pattern = vec!["base/mllm/*.safetensors".to_owned()];
    let health = crate::models::huggingface_cache_health(&repo_root, &pattern);
    assert!(
        !health.installed,
        "a lone AppleDouble sidecar must not satisfy `base/mllm/*.safetensors`"
    );

    // The real shard makes it complete.
    std::fs::write(mllm.join("model.safetensors"), "weights").expect("weights write");
    let fixed = crate::models::huggingface_cache_health(&repo_root, &pattern);
    assert!(
        fixed.installed && !fixed.incomplete,
        "complete once the real shard is present"
    );
}

#[test]
fn filtered_cache_health_reports_absent_filter_as_missing_not_incomplete() {
    // sc-9907: a filter whose files are ENTIRELY absent is cleanly missing, not torn. Only a
    // partially-present filter (some files there, some gone) counts as incomplete/repairable.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let repo_root = temp_dir
        .path()
        .join("models--SceneWorks--z-image-turbo-mlx");
    let snapshot = repo_root.join("snapshots/abc123");
    std::fs::create_dir_all(snapshot.join("q8")).expect("q8 creates");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs creates");
    std::fs::write(repo_root.join("refs/main"), "abc123").expect("ref writes");
    // Only the q8 tier is on disk.
    std::fs::write(snapshot.join("q8/model_index.json"), "{}").expect("q8 file writes");

    // The q4 tier is entirely absent → missing, but NOT incomplete (no false repair prompt).
    let q4 = crate::models::huggingface_cache_health(&repo_root, &["q4/*".to_owned()]);
    assert!(
        !q4.installed && !q4.incomplete,
        "an entirely-absent tier is missing, not incomplete: {q4:?}"
    );

    // The q8 tier is present → installed.
    let q8 = crate::models::huggingface_cache_health(&repo_root, &["q8/*".to_owned()]);
    assert!(
        q8.installed && !q8.incomplete,
        "downloaded tier is installed"
    );
}

#[tokio::test]
async fn quant_matrix_model_with_single_tier_reads_installed_not_incomplete() {
    // sc-9907: a quant-matrix model keeps every tier in ONE shared repo cache. Downloading a single
    // valid tier (here q8, NOT the default q4) previously tripped the top-level default-tier check
    // and surfaced a false "Cached files are incomplete" + Fix button. The card must read installed,
    // never incomplete/repairable, and the per-tier states must show q8 installed / q4+bf16 missing.
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "z_image_turbo",
                "name": "Z-Image Turbo",
                "type": "image",
                "family": "z-image",
                "downloads": [
                  { "provider": "huggingface", "repo": "SceneWorks/z-image-turbo-mlx", "variant": "q4", "default": true, "files": ["q4/*"] },
                  { "provider": "huggingface", "repo": "SceneWorks/z-image-turbo-mlx", "variant": "q8", "files": ["q8/*"] },
                  { "provider": "huggingface", "repo": "SceneWorks/z-image-turbo-mlx", "variant": "bf16", "files": ["bf16/*"] }
                ]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    // Only the non-default q8 tier is on disk in the shared repo snapshot.
    let snapshot = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--SceneWorks--z-image-turbo-mlx/snapshots/abc123");
    std::fs::create_dir_all(snapshot.join("q8")).expect("q8 dir creates");
    std::fs::write(snapshot.join("q8/model_index.json"), "{}").expect("q8 file writes");
    // sc-9909: the real download flow ALSO writes a repo-level completion marker into the app-managed
    // dir (data/models/<repo>). It is tier-agnostic (one marker per repo, no matter which tier was
    // fetched), so the per-tier state must NOT treat its presence as "every tier installed".
    let managed = temp_dir
        .path()
        .join("data/models/SceneWorks__z-image-turbo-mlx");
    std::fs::create_dir_all(&managed).expect("managed dir creates");
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        r#"{ "repo": "SceneWorks/z-image-turbo-mlx" }"#,
    )
    .expect("marker writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "z_image_turbo");
    assert_eq!(models[0]["hasVariantMatrix"], true);
    // Top-level card reads installed — a single valid tier is a complete install, not a repair.
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert_eq!(models[0]["missingRequiredFiles"], json!([]));

    // Per-tier truth: q8 installed; q4 (default) and bf16 cleanly missing, NOT incomplete.
    let variants = models[0]["variants"].as_array().expect("variants array");
    let state_of = |name: &str| {
        variants
            .iter()
            .find(|variant| variant["variant"] == name)
            .unwrap_or_else(|| panic!("variant {name} present"))
            .clone()
    };
    let q8 = state_of("q8");
    assert_eq!(q8["installState"], "installed");
    assert_eq!(q8["cacheState"], "complete");
    for absent in ["q4", "bf16"] {
        let tier = state_of(absent);
        assert_eq!(tier["installState"], "missing", "{absent} installState");
        assert_eq!(tier["cacheState"], "missing", "{absent} cacheState");
    }
}

#[tokio::test]
async fn quant_matrix_empty_cache_skeleton_reads_missing_not_incomplete() {
    // sc-9909: a tier that isn't published upstream resolves ZERO files, leaving an empty HF cache
    // skeleton (bare blobs/, no snapshots) PLUS a stale repo-level completion marker. That must read
    // as a clean not-installed model — NOT a false "installed" (from the marker) and NOT the confusing
    // "Cached files are incomplete: snapshots/<revision>" repair banner.
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "sdxl",
                "name": "SDXL",
                "type": "image",
                "family": "sdxl",
                "downloads": [
                  { "provider": "huggingface", "repo": "SceneWorks/sdxl-base-mlx", "variant": "q4", "default": true, "files": ["q4/*"] },
                  { "provider": "huggingface", "repo": "SceneWorks/sdxl-base-mlx", "variant": "q8", "files": ["q8/*"] },
                  { "provider": "huggingface", "repo": "SceneWorks/sdxl-base-mlx", "variant": "bf16", "files": ["bf16/*"] }
                ]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    // Empty HF cache skeleton: the repo dir exists (bare blobs/) but has NO snapshot revision.
    let repo_root = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--SceneWorks--sdxl-base-mlx");
    std::fs::create_dir_all(repo_root.join("blobs")).expect("blobs dir creates");
    // ...plus a stale repo-level completion marker in the app-managed dir.
    let managed = temp_dir
        .path()
        .join("data/models/SceneWorks__sdxl-base-mlx");
    std::fs::create_dir_all(&managed).expect("managed dir creates");
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        r#"{ "repo": "SceneWorks/sdxl-base-mlx" }"#,
    )
    .expect("marker writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "sdxl");
    // Clean not-installed — no phantom "installed" from the marker, no incomplete/repair banner.
    assert_eq!(models[0]["installState"], "missing");
    assert_eq!(models[0]["cacheState"], "missing");
    assert_eq!(models[0]["repairAvailable"], false);
    for variant in models[0]["variants"].as_array().expect("variants array") {
        assert_eq!(
            variant["installState"], "missing",
            "{} installState",
            variant["variant"]
        );
        assert_eq!(
            variant["cacheState"], "missing",
            "{} cacheState",
            variant["variant"]
        );
    }
}

#[tokio::test]
async fn downloadable_model_catalog_accepts_weightless_component_with_nonstandard_config_name() {
    // Hardening: completeness for weightless auxiliary components is keyed on
    // "the directory exists and holds a file", not a hard-coded config filename.
    // A processor that ships an unexpected config name must still read complete,
    // so future class variants can't re-trigger a permanent false "incomplete".
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    single_model_manifest(
        &temp_dir.path().join("config/manifests"),
        "weightless_model",
        "owner/weightless",
    );
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--weightless/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "SomePipeline",
          "processor": ["transformers", "SomeFutureProcessor"],
          "text_encoder": ["transformers", "T5EncoderModel"],
          "transformer": ["diffusers", "SomeTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    write_complete_weight_bearing_components(&cache_dir);
    // processor dir exists with a config whose name we do NOT special-case.
    let processor_dir = cache_dir.join("processor");
    std::fs::create_dir_all(&processor_dir).expect("processor dir creates");
    std::fs::write(processor_dir.join("processor_config.json"), "{}").expect("processor config");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert_eq!(models[0]["missingRequiredFiles"], json!([]));
}

#[tokio::test]
async fn downloadable_model_catalog_flags_empty_weightless_component_dir() {
    // The hardening must not silently pass everything: a genuinely absent/empty
    // weightless component directory still reports incomplete (partial download).
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    single_model_manifest(
        &temp_dir.path().join("config/manifests"),
        "partial_model",
        "owner/partial",
    );
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--partial/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "SomePipeline",
          "tokenizer": ["transformers", "T5Tokenizer"],
          "text_encoder": ["transformers", "T5EncoderModel"],
          "transformer": ["diffusers", "SomeTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    write_complete_weight_bearing_components(&cache_dir);
    // tokenizer dir is created but left EMPTY (partial download).
    std::fs::create_dir_all(cache_dir.join("tokenizer")).expect("empty tokenizer dir");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["cacheState"], "incomplete");
    assert_eq!(models[0]["repairAvailable"], true);
    assert_eq!(
        models[0]["missingRequiredFiles"],
        json!(["tokenizer/<config>"])
    );
}

#[tokio::test]
async fn model_download_job_forwards_catalog_family_for_worker_reconciliation() {
    // sc-1663: the download job must carry the catalog-declared family so the
    // worker can re-verify the downloaded weights match it (parity with import).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "base_model",
                "name": "Base Model",
                "type": "image",
                "family": "z-image",
                "downloads": [{ "provider": "huggingface", "repo": "owner/model" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, job) = request(
        app,
        "POST",
        "/api/v1/models/base_model/download",
        json!({ "requestedGpu": "auto" }),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "model_download");
    assert_eq!(job["payload"]["modelId"], "base_model");
    assert_eq!(job["payload"]["family"], "z-image");
}

#[test]
fn model_download_and_co_requisite_helpers_partition_downloads() {
    // sc-9696: a co-requisite (fetch-all dependency) must never be chosen as the primary/tier
    // download, and must be enumerable separately for the download job + install-state gating.
    let model = json!({
        "id": "pid_qwenimage",
        "downloads": [
            { "provider": "huggingface", "repo": "SceneWorks/pid-qwenimage" },
            { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it", "coRequisite": true }
        ]
    });
    assert_eq!(
        crate::model_download(&model).expect("primary resolves")["repo"],
        "SceneWorks/pid-qwenimage"
    );
    let co_requisites = crate::model_co_requisite_downloads(&model);
    assert_eq!(co_requisites.len(), 1);
    assert_eq!(co_requisites[0]["repo"], "SceneWorks/gemma-2-2b-it");

    // Ordering-independent: a co-requisite listed FIRST still never wins the primary slot.
    let reordered = json!({
        "id": "pid_qwenimage",
        "downloads": [
            { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it", "coRequisite": true },
            { "provider": "huggingface", "repo": "SceneWorks/pid-qwenimage" }
        ]
    });
    assert_eq!(
        crate::model_download(&reordered).expect("primary resolves")["repo"],
        "SceneWorks/pid-qwenimage"
    );
}

#[tokio::test]
async fn model_download_job_enqueues_co_requisite_dependencies() {
    // sc-9696: installing a model with a co-requisite download (e.g. the PiD decoder's shared
    // gemma-2-2b-it caption encoder) must queue a SECOND download job for the dependency — else it
    // is never fetched and the feature silently no-ops (PiD → native VAE). The co-requisite job
    // must NOT carry the model's family (it is a different artifact than the primary checkpoint).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "pid_qwenimage",
                "name": "PiD Decoder",
                "type": "utility",
                "family": "pid",
                "downloads": [
                  { "provider": "huggingface", "repo": "SceneWorks/pid-qwenimage" },
                  { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it", "coRequisite": true }
                ]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    write_empty_sibling_manifests(&config_dir);

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, primary) = request(
        app.clone(),
        "POST",
        "/api/v1/models/pid_qwenimage/download",
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // The returned job is the primary (its id is what the download UI tracks).
    assert_eq!(primary["payload"]["repo"], "SceneWorks/pid-qwenimage");
    assert_eq!(primary["payload"]["family"], "pid");

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    let download_jobs = jobs
        .as_array()
        .expect("jobs is an array")
        .iter()
        .filter(|job| job["type"] == "model_download")
        .collect::<Vec<_>>();
    assert_eq!(
        download_jobs.len(),
        2,
        "the primary and its one co-requisite must each get a download job"
    );
    let co_requisite = download_jobs
        .iter()
        .find(|job| job["payload"]["repo"] == "SceneWorks/gemma-2-2b-it")
        .expect("a co-requisite download job is enqueued");
    assert!(
        co_requisite["payload"].get("family").map_or(true, Value::is_null),
        "a co-requisite job must not carry the model family (different artifact than the checkpoint)"
    );
}

#[tokio::test]
async fn model_download_job_forwards_pinned_revision_for_co_requisite() {
    // sc-13541: a co-requisite whose weight the runtime resolves via a pinned-SHA `hf_get_pinned`
    // (chatterbox_tts's ve/perth) must have its `revision` forwarded to the download job, so the worker
    // fetches that exact commit into `snapshots/<sha>/` — where the resolver reads offline. The planner
    // historically dropped `revision` and the worker defaulted to `main`, which would populate the wrong
    // snapshot. The primary download declares NO revision and must keep the `main` default (no
    // `revision` key), so this change is additive and can't perturb the other seeded models.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "chatterbox_tts",
                "name": "Chatterbox",
                "type": "audio",
                "family": "chatterbox",
                "downloads": [
                  { "provider": "huggingface", "repo": "ResembleAI/chatterbox", "files": ["t3_cfg.safetensors"] },
                  { "provider": "huggingface", "repo": "ResembleAI/chatterbox", "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", "coRequisite": true, "files": ["ve.safetensors"] },
                  { "provider": "huggingface", "repo": "SceneWorks/perth-implicit", "revision": "80b60f9caead09b8d3b512bda0b24038f28c08ec", "coRequisite": true, "files": ["perth_implicit.safetensors"] }
                ]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    write_empty_sibling_manifests(&config_dir);

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, primary) = request(
        app.clone(),
        "POST",
        "/api/v1/models/chatterbox_tts/download",
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // The primary download stays on main: no `revision` key is forwarded.
    assert_eq!(primary["payload"]["repo"], "ResembleAI/chatterbox");
    assert!(
        primary["payload"]
            .get("revision")
            .map_or(true, Value::is_null),
        "the primary download must not carry a pinned revision (defaults to main)"
    );

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    let download_jobs = jobs
        .as_array()
        .expect("jobs is an array")
        .iter()
        .filter(|job| job["type"] == "model_download")
        .collect::<Vec<_>>();
    assert_eq!(
        download_jobs.len(),
        3,
        "the primary plus its two pinned companion co-requisites each get a download job"
    );
    for (repo, files, sha) in [
        (
            "ResembleAI/chatterbox",
            "ve.safetensors",
            "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        ),
        (
            "SceneWorks/perth-implicit",
            "perth_implicit.safetensors",
            "80b60f9caead09b8d3b512bda0b24038f28c08ec",
        ),
    ] {
        let job = download_jobs
            .iter()
            .find(|job| job["payload"]["repo"] == repo && job["payload"]["files"][0] == files)
            .unwrap_or_else(|| panic!("a download job for the {repo} co-requisite is enqueued"));
        assert_eq!(
            job["payload"]["revision"], sha,
            "the {repo} co-requisite job must forward its pinned revision so the worker fetches the \
             exact snapshot the runtime resolver reads offline"
        );
    }
}

#[tokio::test]
async fn model_catalog_gates_install_state_on_co_requisite() {
    // sc-9696: the entry is "installed" only when the primary AND every co-requisite are cached.
    // Primary present but the gemma-2-2b-it caption encoder missing → not-installed + repairable, so
    // the web PiD toggle stays hidden and the user still has a path to fetch the missing dependency.
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "pid_qwenimage",
                "name": "PiD Decoder",
                "type": "utility",
                "family": "pid",
                "downloads": [
                  { "provider": "huggingface", "repo": "SceneWorks/pid-qwenimage" },
                  { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it", "coRequisite": true }
                ]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    write_empty_sibling_manifests(&config_dir);

    let hub = temp_dir.path().join("data/cache/huggingface/hub");
    // Primary present: a payload file makes a non-diffusers snapshot resolve as installed.
    let primary_snapshot = hub.join("models--SceneWorks--pid-qwenimage/snapshots/rev1");
    std::fs::create_dir_all(&primary_snapshot).expect("primary snapshot dir creates");
    std::fs::write(
        primary_snapshot.join("pid_qwenimage_2kto4k.safetensors"),
        "weights",
    )
    .expect("primary weights write");

    // Case A: gemma co-requisite absent → not installed, repairable, and surfaced as missing.
    let (status, models) = request(
        create_app(test_settings(&temp_dir)).expect("app creates"),
        "GET",
        "/api/v1/models",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "pid_qwenimage");
    assert_eq!(
        models[0]["installState"], "missing",
        "primary present but co-requisite gemma missing → not installed"
    );
    assert_eq!(models[0]["cacheState"], "incomplete");
    assert_eq!(models[0]["repairAvailable"], true);
    assert!(
        models[0]["missingRequiredFiles"]
            .as_array()
            .expect("missingRequiredFiles is an array")
            .iter()
            .any(|entry| entry
                .as_str()
                .is_some_and(|value| value.contains("gemma-2-2b-it"))),
        "the missing co-requisite must be surfaced in missingRequiredFiles"
    );

    // Case B: install the gemma co-requisite too → the entry now reports installed.
    let gemma_snapshot = hub.join("models--SceneWorks--gemma-2-2b-it/snapshots/rev1");
    std::fs::create_dir_all(&gemma_snapshot).expect("gemma snapshot dir creates");
    std::fs::write(gemma_snapshot.join("config.json"), "{}").expect("gemma config write");
    std::fs::write(gemma_snapshot.join("model.safetensors"), "weights")
        .expect("gemma weights write");

    let (status, models) = request(
        create_app(test_settings(&temp_dir)).expect("app creates"),
        "GET",
        "/api/v1/models",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        models[0]["installState"], "installed",
        "primary + co-requisite both cached → installed"
    );
    assert_eq!(models[0]["cacheState"], "complete");
}

#[tokio::test]
async fn lora_catalog_uses_huggingface_cache_install_state() {
    let _env = isolate_hf_cache(); // resolve the HF cache under the tempdir, never a dev's real cache (sc-13835)
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [{
                "id": "ltx_ic_union",
                "name": "LTX IC Union",
                "family": "ltx-video",
                "icLora": true,
                "conditioningRole": "ic_lora",
                "compatibility": { "families": ["ltx-video"] },
                "source": {
                  "provider": "huggingface",
                  "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                  "file": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
                }
              }]
            }
            "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");
    let stale_cache_file = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/snapshots/aaa111/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors");
    std::fs::create_dir_all(
        stale_cache_file
            .parent()
            .expect("stale cache file has parent"),
    )
    .expect("stale hf cache creates");
    std::fs::write(&stale_cache_file, b"stale-lora").expect("stale lora cache writes");
    let cache_file = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/snapshots/zzz999/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors");
    std::fs::create_dir_all(cache_file.parent().expect("cache file has parent"))
        .expect("hf cache creates");
    std::fs::write(&cache_file, b"lora").expect("lora cache writes");
    let refs_main = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/refs/main");
    std::fs::create_dir_all(refs_main.parent().expect("refs main has parent"))
        .expect("refs dir creates");
    std::fs::write(&refs_main, b"zzz999").expect("refs main writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(loras[0]["id"], "ltx_ic_union");
    assert_eq!(loras[0]["icLora"], true);
    assert_eq!(loras[0]["conditioningRole"], "ic_lora");
    assert_eq!(loras[0]["installState"], "installed");
    assert_eq!(
        std::path::PathBuf::from(loras[0]["installedPath"].as_str().expect("installed path")),
        cache_file
    );
}

#[test]
fn lora_artifact_paths_exclude_shared_huggingface_cache_files() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let cache_file = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--repo/snapshots/abc123/lora.safetensors");
    let lora = json!({
        "id": "hf_lora",
        "installedPath": cache_file.display().to_string(),
        "source": {
            "provider": "huggingface",
            "repo": "owner/repo",
            "file": "lora.safetensors"
        }
    });

    assert!(lora_artifact_paths(&lora, temp_dir.path()).is_empty());
}

#[tokio::test]
async fn catalog_delete_routes_remove_manifest_entries_and_owned_artifacts() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let model_dir = temp_dir.path().join("data/models/imports/delete_me");
    let lora_dir = temp_dir.path().join("data/loras/delete_style");
    std::fs::create_dir_all(&model_dir).expect("model dir creates");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("marker writes");
    std::fs::write(lora_dir.join("adapter.safetensors"), b"lora").expect("lora writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        format!(
            r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "delete_me",
                    "name": "Delete Me",
                    "type": "image",
                    "family": "z-image",
                    "paths": {{ "model": "{}" }}
                  }}]
                }}"#,
            model_dir.display().to_string().replace('\\', "\\\\")
        ),
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        format!(
            r#"{{
                  "schemaVersion": 1,
                  "loras": [{{
                    "id": "delete_style",
                    "name": "Delete Style",
                    "family": "z-image",
                    "source": {{ "provider": "local", "path": "{}" }}
                  }}]
                }}"#,
            lora_dir.display().to_string().replace('\\', "\\\\")
        ),
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "moody",
                  "name": "Moody",
                  "workflow": "text_to_image",
                  "model": "delete_me",
                  "loras": [{ "id": "delete_style" }]
                }
              ]
            }
            "#,
    )
    .expect("user presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    // permanent=true keeps the assertions deterministic (the default move-to-OS-trash
    // path depends on the host having a usable recycle bin/trash).
    let (model_status, model_delete) = request(
        app.clone(),
        "DELETE",
        "/api/v1/models/delete_me?permanent=true",
        Value::Null,
    )
    .await;
    assert_eq!(model_status, StatusCode::OK);
    assert_eq!(model_delete["removedManifestEntry"], true);
    assert_eq!(model_delete["removedLocalArtifacts"], true);
    assert!(model_delete["warnings"][0]
        .as_str()
        .is_some_and(|warning| warning.contains("Moody")));
    assert!(!model_dir.exists());
    let models_manifest =
        std::fs::read_to_string(config_dir.join("user.models.jsonc")).expect("models reads");
    assert!(!models_manifest.contains("delete_me"));

    let (lora_status, lora_delete) = request(
        app.clone(),
        "DELETE",
        "/api/v1/loras/delete_style?scope=global&permanent=true",
        Value::Null,
    )
    .await;
    assert_eq!(lora_status, StatusCode::OK);
    assert_eq!(lora_delete["removedManifestEntry"], true);
    assert_eq!(lora_delete["removedLocalArtifacts"], true);
    assert!(lora_delete["warnings"][0]
        .as_str()
        .is_some_and(|warning| warning.contains("Moody")));
    assert!(!lora_dir.exists());
    let loras_manifest =
        std::fs::read_to_string(config_dir.join("user.loras.jsonc")).expect("loras reads");
    assert!(!loras_manifest.contains("delete_style"));

    let (models_status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(models_status, StatusCode::OK);
    assert_eq!(models.as_array().expect("models array").len(), 0);
    let (loras_status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(loras_status, StatusCode::OK);
    assert_eq!(loras.as_array().expect("loras array").len(), 0);
}

#[test]
fn model_download_size_helpers_match_contract_shapes() {
    let siblings = json!([
        { "rfilename": "model-00001.safetensors", "size": 100 },
        { "rfilename": "model-00002.safetensors", "size": "200" },
        { "rfilename": "README.md", "size": 50 },
        { "rfilename": "unknown.bin" }
    ]);
    let siblings = siblings.as_array().expect("siblings array");
    assert_eq!(
        crate::download_size_from_siblings(siblings, &["*.safetensors".to_owned()]),
        Some(300)
    );
    assert_eq!(
        crate::download_size_from_siblings(siblings, &["*.ckpt".to_owned()]),
        None
    );
    assert_eq!(crate::json_size_to_u64(&json!("200.5")), None);
    assert_eq!(crate::format_bytes(0), "0 B");
    assert_eq!(crate::format_bytes(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(
        crate::manifest_download_size_bytes(
            &json!({ "downloads": [] }),
            &json!({ "estimatedSizeBytes": "4096" })
        ),
        Some(4096)
    );
    assert_eq!(
        crate::manifest_download_size_bytes(&json!({ "sizeBytes": 2048 }), &json!({})),
        Some(2048)
    );
    assert_eq!(
        crate::quote_huggingface_repo("owner/model name"),
        "owner/model%20name"
    );
    assert!(crate::model_download(&json!({
        "downloads": [{ "repo": "owner/model" }]
    }))
    .is_none());
    assert_eq!(
            crate::model_download(&json!({
                "downloads": [
                    { "provider": "huggingface", "repo": "owner/fallback", "estimatedSizeBytes": 1024 },
                    { "provider": "huggingface", "repo": "owner/default", "default": true, "estimatedSizeBytes": 4096 }
                ]
            }))
            .and_then(|download| download.get("repo").and_then(Value::as_str).map(str::to_owned)),
            Some("owner/default".to_owned())
        );
    let mut cache = crate::ModelSizeCache::default();
    let key = ("owner/model".to_owned(), vec!["*.safetensors".to_owned()]);
    cache.insert(key.clone(), 300);
    assert_eq!(cache.get(&key), Some(Some(300)));
    // sc-4169: failed estimates are negative-cached — `Some(None)` tells the
    // caller to skip the network — and expire after the TTL (a cache miss).
    let failed = ("owner/offline".to_owned(), vec!["*.safetensors".to_owned()]);
    cache.insert_failure(failed.clone());
    assert_eq!(cache.get(&failed), Some(None));
    cache.insert_failure_expiring_at(failed.clone(), std::time::Instant::now());
    assert_eq!(
        cache.get(&failed),
        None,
        "expired negative entry must be a miss"
    );
    // A later successful estimate replaces a cached failure.
    cache.insert_failure(failed.clone());
    cache.insert(failed.clone(), 700);
    assert_eq!(cache.get(&failed), Some(Some(700)));
    assert!(crate::allow_pattern_matches(
        "model-7.safetensors",
        &["model-[0-9].safetensors".to_owned()]
    ));
    if cfg!(windows) {
        assert!(crate::allow_pattern_matches(
            "Model.SAFETENSORS",
            &["*.safetensors".to_owned()]
        ));
    }
}

#[test]
fn platform_tagged_downloads_resolve_per_os() {
    // A video model that ships a native MLX-convert checkpoint (macOS) and a diffusers/torch
    // checkpoint (Windows/Linux) for the same model id (sc-3240).
    let model = json!({
        "downloads": [
            { "provider": "huggingface", "repo": "Wan-AI/Wan2.2-TI2V-5B", "platforms": ["macos"] },
            { "provider": "huggingface", "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "platforms": ["windows", "linux"] }
        ]
    });
    let resolved_repo = |model: &Value| {
        crate::model_download(model).and_then(|download| {
            download
                .get("repo")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
    };

    // macOS keeps the native MLX-convert source...
    let mut mac = model.clone();
    crate::retain_downloads_for_os(&mut mac, "macos");
    assert_eq!(
        resolved_repo(&mac),
        Some("Wan-AI/Wan2.2-TI2V-5B".to_owned())
    );

    // ...Windows/Linux keep the diffusers/torch repo.
    for os in ["windows", "linux"] {
        let mut other = model.clone();
        crate::retain_downloads_for_os(&mut other, os);
        assert_eq!(
            resolved_repo(&other),
            Some("Wan-AI/Wan2.2-TI2V-5B-Diffusers".to_owned()),
            "os={os}"
        );
    }

    // Untagged single-repo models are untouched on every OS.
    let mut agnostic = json!({
        "downloads": [{ "provider": "huggingface", "repo": "owner/model" }]
    });
    crate::retain_downloads_for_os(&mut agnostic, "macos");
    assert_eq!(
        agnostic["downloads"].as_array().map(Vec::len),
        Some(1),
        "agnostic downloads must not be filtered"
    );
}

#[test]
fn lora_family_filter_shapes_match_contract_fallbacks() {
    let shapes = [
        json!({ "families": ["z-image"] }),
        json!({ "compatibleFamilies": ["z-image"] }),
        json!({ "modelFamilies": ["z-image"] }),
        json!({ "compatibility": { "families": ["z-image"] } }),
        json!({ "family": "z-image" }),
    ];
    for lora in shapes {
        assert_eq!(crate::lora_families(&lora), vec!["z-image".to_owned()]);
    }
}

#[test]
fn builtin_manifest_registers_the_prompt_refine_model() {
    // sc-5605: the native prompt_refine worker (prompt_refine_jobs.rs) resolves an
    // already-cached HF snapshot via huggingface_snapshot_dir and does NOT auto-download
    // (unlike the retired Python PromptRefiner's from_pretrained). The refine LLM must
    // therefore be a provisionable catalog artifact so Model Manager can download it into
    // the HF cache the worker reads from. This guards the real manifest entry against
    // accidental removal and against the repo string drifting from the worker's
    // DEFAULT_REFINE_MODEL.
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/manifests/builtin.models.jsonc");
    let raw = std::fs::read_to_string(&manifest_path).expect("read builtin.models.jsonc");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&raw)).expect("parse builtin.models.jsonc");
    let model = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry["id"] == "prompt_refine_anubis_8b")
        .expect("prompt_refine_anubis_8b is registered in the catalog");
    // A non-generation utility entry (mirrors the upscalers): absent from the image/video
    // studio pickers, present + downloadable in Model Manager.
    assert_eq!(model["type"], "utility");
    let download = model["downloads"]
        .as_array()
        .and_then(|downloads| downloads.first())
        .expect("a download entry");
    assert_eq!(download["provider"], "huggingface");
    // Must match the worker's DEFAULT_REFINE_MODEL (prompt_refine_jobs.rs) — the string the
    // worker passes to huggingface_snapshot_dir.
    assert_eq!(
        download["repo"], "TheDrummer/Anubis-Mini-8B-v1",
        "manifest repo must match the worker's DEFAULT_REFINE_MODEL"
    );
    // The catalog install-state path must resolve the same HF repo the worker does.
    assert_eq!(
        model["paths"]["model"],
        "${HF_CACHE}/TheDrummer/Anubis-Mini-8B-v1"
    );
}

#[test]
fn builtin_manifest_registers_the_wan_vace_fun_model() {
    // sc-3458 (epic 3456): Wan2.2 VACE-Fun A14B is a first-class native video model. Guard the
    // catalog entry against accidental removal/drift and assert the honest, native-first shape:
    // the wan-video adapter, ONLY the validated replace_person capability (no generic T2V/I2V on
    // this control checkpoint), the diffusers-conversion download repo (NOT the raw VideoX-Fun
    // alibaba-pai repo, which does not load), an MLX block, and NO Torch GGUF quantization.
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/manifests/builtin.models.jsonc");
    let raw = std::fs::read_to_string(&manifest_path).expect("read builtin.models.jsonc");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&raw)).expect("parse builtin.models.jsonc");
    let model = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry["id"] == "wan_2_2_vace_fun_14b")
        .expect("wan_2_2_vace_fun_14b is registered in the catalog");
    assert_eq!(model["type"], "video");
    assert_eq!(model["family"], "wan-video");
    assert_eq!(model["adapter"], "wan_video");

    // Only the validated VACE mode is exposed; generic T2V/I2V are NOT (control checkpoint).
    let caps: Vec<&str> = model["capabilities"]
        .as_array()
        .expect("capabilities array")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(
        caps,
        vec!["replace_person"],
        "expose only the validated VACE mode"
    );

    // Per-platform download split (sc-8613): macOS pulls the native MLX VACE-Fun checkpoint;
    // Windows/Linux pull the DIFFERENT candle Wan2.1-VACE-14B diffusers tree the candle `wan_vace`
    // provider reads (CANDLE_WAN_VACE_REPO). Both are diffusers-loadable conversions, never the raw
    // VideoX-Fun upstream. Every OS must have exactly one install path.
    let downloads = model["downloads"].as_array().expect("downloads array");
    let download_for = |os: &str| {
        downloads
            .iter()
            .find(|download| {
                download["platforms"]
                    .as_array()
                    .map(|platforms| platforms.iter().any(|p| p.as_str() == Some(os)))
                    .unwrap_or(false)
            })
            .unwrap_or_else(|| panic!("a download covering {os}"))
    };
    let macos = download_for("macos");
    assert_eq!(macos["provider"], "huggingface");
    assert_eq!(macos["repo"], "linoyts/Wan2.2-VACE-Fun-14B-diffusers");
    let windows = download_for("windows");
    assert_eq!(windows["repo"], "Wan-AI/Wan2.1-VACE-14B-diffusers");
    assert_eq!(
        download_for("linux")["repo"],
        "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "Linux rides the same candle checkpoint as Windows"
    );
    for download in downloads {
        assert_eq!(download["provider"], "huggingface");
        assert_ne!(
            download["repo"], "alibaba-pai/Wan2.2-VACE-Fun-A14B",
            "must not point at the raw VideoX-Fun repo (it does not load via diffusers/native)"
        );
    }

    // Native-only: an MLX block, and NO Torch GGUF quantization variants.
    assert!(model["mlx"].is_object(), "native MLX engine block present");
    assert!(
        model["mlx"]["minMemoryGb"].as_u64().unwrap_or(0) >= 64,
        "dual 14B needs a substantial memory floor"
    );
    assert!(
        model.get("quantization").is_none(),
        "native-first: no Torch GGUF quantization block"
    );
}

#[test]
fn builtin_manifest_registers_wan_a14b_lightning_corequisite() {
    // sc-10030 (epic 8506): both A14B MoE video models use the 4-step lightx2v Lightning distill by
    // default. As of sc-10047 Lightning is a DEFAULT-ON toggle (`advanced.lightning`) rather than
    // mandatory — a job can opt out and run the native multi-step CFG recipe — but the default path
    // still applies the high/low pair (wan_sampling → 4-step/CFG-off; resolve_wan_adapters prepends the
    // pair when the toggle is on). So it must still install as a macOS `coRequisite` so the model
    // manager provisions it and install_state gates on it — without this the default gen errors "not
    // downloaded — fetch it via the model manager". The subdir is per-architecture, NOT cross-compatible.
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/manifests/builtin.models.jsonc");
    let raw = std::fs::read_to_string(&manifest_path).expect("read builtin.models.jsonc");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&raw)).expect("parse builtin.models.jsonc");
    let models = manifest["models"].as_array().expect("models array");

    // (engine id, expected per-architecture Lightning subdir prefix)
    let cases = [
        (
            "wan_2_2_t2v_14b",
            "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        ),
        (
            "wan_2_2_i2v_14b",
            "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        ),
    ];
    for (model_id, subdir) in cases {
        let model = models
            .iter()
            .find(|entry| entry["id"] == model_id)
            .unwrap_or_else(|| panic!("{model_id} is registered in the catalog"));
        let downloads = model["downloads"].as_array().expect("downloads array");
        let lightning = downloads
            .iter()
            .find(|download| download["coRequisite"] == Value::Bool(true))
            .unwrap_or_else(|| panic!("{model_id} declares a Lightning coRequisite"));
        assert_eq!(lightning["provider"], "huggingface");
        assert_eq!(
            lightning["repo"], "lightx2v/Wan2.2-Lightning",
            "{model_id} coRequisite points at the lightx2v Lightning repo"
        );
        // macOS-only (the native MLX path is Mac; Windows/Linux use the torch adapter).
        let platforms: Vec<&str> = lightning["platforms"]
            .as_array()
            .expect("coRequisite platforms array")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(
            platforms,
            vec!["macos"],
            "{model_id} Lightning is macOS-only"
        );
        // Exactly the per-architecture high/low pair — nothing cross-compatible, no preview assets.
        let files: Vec<&str> = lightning["files"]
            .as_array()
            .expect("coRequisite files array")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(
            files,
            vec![
                format!("{subdir}/high_noise_model.safetensors").as_str(),
                format!("{subdir}/low_noise_model.safetensors").as_str(),
            ],
            "{model_id} fetches exactly its per-architecture Lightning pair"
        );
        // A coRequisite is never a selectable quant tier.
        assert!(
            lightning.get("variant").is_none(),
            "{model_id} Lightning coRequisite is not a quant tier"
        );
    }
}

#[test]
fn builtin_manifest_pins_chatterbox_companion_corequisites() {
    // sc-13541 (epic 13400 / E1 follow-up): the chatterbox_tts generator resolves two companion
    // weights at generate() time via pinned-SHA `hf_get_pinned` fetches, NOT from its snapshot dir —
    // the ve.safetensors speaker embedder (ResembleAI/chatterbox @ 5bb1f6ee…) and the PerTh
    // watermarker (SceneWorks/perth-implicit @ 80b60f9c…; generate() ALWAYS watermarks, so a missing
    // PerTh weight is a hard error). Both must install as `coRequisite` downloads pinned to the EXACT
    // full 40-hex commit SHA the resolver uses: `hf_get_pinned` refuses a branch/tag/short SHA and
    // reads `models--<org>--<name>/snapshots/<sha>/`, so a main-branch predownload lands in the wrong
    // snapshot and offline generation fails. This test guards both the presence AND the exact pinned
    // revisions (a SHA drift here silently reintroduces the offline gap this story closed).
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/manifests/builtin.models.jsonc");
    let raw = std::fs::read_to_string(&manifest_path).expect("read builtin.models.jsonc");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&raw)).expect("parse builtin.models.jsonc");
    let models = manifest["models"].as_array().expect("models array");
    let chatterbox = models
        .iter()
        .find(|entry| entry["id"] == "chatterbox_tts")
        .expect("chatterbox_tts is registered in the catalog");
    let downloads = chatterbox["downloads"].as_array().expect("downloads array");

    // The primary download (the three files the generator loads from its snapshot dir) is NOT a
    // co-requisite. Under the F-029 pin migration (sc-13685) it now carries the SAME pinned SHA the
    // inference runtime self-fetches — `5bb1f6ee…`, the exact commit the `voice_embedding` coReq in
    // this same ResembleAI/chatterbox repo already pins — so an offline install lands the primary in
    // the snapshot dir the resolver reads. A SHA drift here reopens the offline gap this story closed.
    let primary = downloads
        .iter()
        .find(|download| download.get("coRequisite").and_then(Value::as_bool) != Some(true))
        .expect("chatterbox_tts has a primary download");
    assert_eq!(primary["repo"], "ResembleAI/chatterbox");
    let primary_revision = primary["revision"]
        .as_str()
        .expect("the primary chatterbox download pins a revision (F-029 migration, sc-13685)");
    assert_eq!(
        primary_revision, "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        "the primary chatterbox download must pin the F-029 revision (sc-13685) the runtime self-fetches"
    );

    // (repo, file, expected full-SHA revision) for each companion co-requisite. The SHAs are the exact
    // pins in the inference resolvers at the SceneWorks-pinned inference commit:
    //   ve   -> candle_audio_chatterbox_ve::{HUB_REPO, HUB_REVISION}      (crates/audio/candle-audio-chatterbox-ve/src/model.rs)
    //   perth-> candle_audio_chatterbox::perth::{PERTH_HUB_REPO, PERTH_HUB_REVISION} (crates/audio/candle-audio-chatterbox/src/perth.rs)
    // This test hardcodes them (it cannot link the git-dep consts), so a drift between the manifest and
    // the resolver still needs the offline DoD smoke (voiceclone_smoke.rs) to catch it end to end.
    let cases = [
        (
            "ResembleAI/chatterbox",
            "ve.safetensors",
            "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        ),
        (
            "SceneWorks/perth-implicit",
            "perth_implicit.safetensors",
            "80b60f9caead09b8d3b512bda0b24038f28c08ec",
        ),
    ];
    let co_requisites: Vec<&Value> = downloads
        .iter()
        .filter(|download| download.get("coRequisite").and_then(Value::as_bool) == Some(true))
        .collect();
    assert_eq!(
        co_requisites.len(),
        cases.len(),
        "chatterbox_tts declares exactly the ve + PerTh companion co-requisites"
    );
    for (repo, file, sha) in cases {
        let entry = co_requisites
            .iter()
            .find(|download| download["repo"] == repo && download["files"][0] == file)
            .unwrap_or_else(|| panic!("chatterbox_tts declares the {repo} / {file} co-requisite"));
        assert_eq!(entry["provider"], "huggingface");
        let files: Vec<&str> = entry["files"]
            .as_array()
            .expect("co-requisite files array")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(
            files,
            vec![file],
            "{repo} co-requisite fetches exactly the one companion weight"
        );
        let revision = entry["revision"]
            .as_str()
            .unwrap_or_else(|| panic!("{repo} co-requisite pins a revision"));
        assert_eq!(
            revision, sha,
            "{repo} co-requisite must pin the exact SHA the runtime's hf_get_pinned resolves"
        );
        // hf_get_pinned rejects anything but a full 40-hex commit SHA, so a short/branch revision would
        // resolve online but hard-fail offline — assert the shape the resolver requires.
        assert_eq!(
            revision.len(),
            40,
            "{repo} revision must be a full commit SHA"
        );
        assert!(
            revision.chars().all(|c| c.is_ascii_hexdigit()),
            "{repo} revision must be a hex commit SHA"
        );
        // A co-requisite is never a selectable quant tier, and it must be fetched on every platform the
        // model supports (candle-native everywhere), matching the primary download's platform set.
        assert!(
            entry.get("variant").is_none(),
            "{repo} co-requisite is not a quant tier"
        );
        let platforms: Vec<&str> = entry["platforms"]
            .as_array()
            .expect("co-requisite platforms array")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(
            platforms,
            vec!["macos", "windows", "linux"],
            "{repo} co-requisite installs on every platform"
        );
    }
}

#[test]
fn builtin_manifest_registers_the_joycaption_model() {
    // sc-5620: the native captioner (caption_jobs.rs, the training_caption job) resolves an
    // already-cached HF snapshot via the same resolve_app_managed_model_dir seam and does NOT
    // auto-download. JoyCaption must be a provisionable catalog artifact (same gap as sc-5605's
    // prompt_refine). Guards the entry + repo == caption_jobs::JOY_CAPTION_MODEL + the cache path.
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/manifests/builtin.models.jsonc");
    let raw = std::fs::read_to_string(&manifest_path).expect("read builtin.models.jsonc");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&raw)).expect("parse builtin.models.jsonc");
    let model = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry["id"] == "joycaption_beta_one")
        .expect("joycaption_beta_one is registered in the catalog");
    assert_eq!(model["type"], "utility");
    let download = model["downloads"]
        .as_array()
        .and_then(|downloads| downloads.first())
        .expect("a download entry");
    assert_eq!(download["provider"], "huggingface");
    // Must match the worker's JOY_CAPTION_MODEL (caption_jobs.rs) — the string the worker passes
    // to huggingface_snapshot_dir.
    assert_eq!(
        download["repo"], "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "manifest repo must match the worker's JOY_CAPTION_MODEL"
    );
    assert_eq!(
        model["paths"]["model"],
        "${HF_CACHE}/fancyfeast/llama-joycaption-beta-one-hf-llava"
    );
}

#[tokio::test]
async fn lora_import_records_trigger_words_and_notes() {
    // epic 10328: the import request carries trigger keywords + usage notes, and both
    // land on the queued manifest entry the worker later persists.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "repo": "owner/lora",
            "name": "Keyworded LoRA",
            "files": ["adapter.safetensors"],
            "triggerWords": ["sksStyle", "neon"],
            "notes": "Combine sksStyle with neon; keep weight <= 0.7."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let entry = &job["payload"]["manifestEntry"];
    assert_eq!(entry["triggerWords"], json!(["sksStyle", "neon"]));
    assert_eq!(
        entry["notes"],
        "Combine sksStyle with neon; keep weight <= 0.7."
    );
}

#[tokio::test]
async fn update_lora_edits_trigger_words_and_notes() {
    // epic 10328: PATCH /api/v1/loras/:id edits keywords/notes after import, supports
    // partial updates, and the edit surfaces in the catalog listing.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let settings = test_settings(&temp_dir);
    let manifest_dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir");
    std::fs::write(
        manifest_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [
            { "id": "my_lora", "name": "My LoRA", "family": "z-image",
              "triggerWords": ["old"], "notes": "",
              "source": { "provider": "local", "path": "loras/my_lora.safetensors" } }
        ] }"#,
    )
    .expect("seed manifest");

    let app = create_app(settings).expect("app creates");

    // Full update.
    let (status, updated) = request(
        app.clone(),
        "PATCH",
        "/api/v1/loras/my_lora?scope=global",
        json!({ "triggerWords": ["fresh", "tokens"], "notes": "use at 0.8" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["triggerWords"], json!(["fresh", "tokens"]));
    assert_eq!(updated["notes"], "use at 0.8");

    // Partial update (only notes) leaves the keywords untouched.
    let (status, updated) = request(
        app.clone(),
        "PATCH",
        "/api/v1/loras/my_lora?scope=global",
        json!({ "notes": "revised note" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["triggerWords"], json!(["fresh", "tokens"]));
    assert_eq!(updated["notes"], "revised note");

    // The edit is reflected in GET /api/v1/loras.
    let (status, loras) = request(app, "GET", "/api/v1/loras", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    let entry = loras
        .as_array()
        .expect("catalog is an array")
        .iter()
        .find(|item| item["id"] == "my_lora")
        .expect("seeded LoRA present");
    assert_eq!(entry["triggerWords"], json!(["fresh", "tokens"]));
    assert_eq!(entry["notes"], "revised note");
}

/// epic 10451 / sc-10452: with an operator-configured external root, the LoRAs in a
/// ComfyUI `models/loras` tree are surfaced by `GET /api/v1/loras` — read in place,
/// never copied — and are read-only: not removable, and `DELETE` refuses them. We
/// borrowed the user's file; we must not offer to destroy it.
#[tokio::test]
async fn external_root_loras_are_listed_read_only() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let comfy_root = temp_dir.path().join("ComfyUI").join("models");
    // Nested, like the real tree (`loras/Wan/…`).
    let adapter = comfy_root
        .join("loras")
        .join("Wan")
        .join("detailz-wan.safetensors");
    write_comfy_wan_adapter(&adapter);

    let mut settings = test_settings(&temp_dir);
    settings.external_model_roots = vec![comfy_root];
    let app = create_app(settings).expect("app creates");

    let (status, loras) = request(app.clone(), "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let external = loras
        .as_array()
        .expect("loras array")
        .iter()
        .find(|lora| lora["scope"] == "external")
        .expect("the ComfyUI adapter is surfaced");

    assert_eq!(external["name"], "Wan/detailz-wan");
    assert_eq!(external["family"], "wan-video");
    assert_eq!(external["installState"], "installed");
    // Read in place: the catalog points at the operator's own file, not a copy.
    assert_eq!(
        external["installedPath"].as_str().expect("installedPath"),
        adapter
            .canonicalize()
            .expect("canonicalize")
            .display()
            .to_string()
    );
    // Never offer to delete a file we do not own.
    assert_eq!(external["removable"], false);

    let id = external["id"].as_str().expect("id");
    assert!(id.starts_with("external_"), "ids are namespaced: {id}");

    let (status, _) = request(
        app,
        "DELETE",
        &format!("/api/v1/loras/{id}?scope=external"),
        Value::Null,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "deleting an external LoRA must be refused"
    );
}

/// The feature is off unless an operator opts in: with no external roots configured
/// (the default), the catalog is exactly what the manifests declare.
#[tokio::test]
async fn no_external_roots_leaves_the_lora_catalog_unchanged() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let comfy_root = temp_dir.path().join("ComfyUI").join("models");
    write_comfy_wan_adapter(&comfy_root.join("loras").join("ignored.safetensors"));

    // Settings default to no external roots; the tree above is simply not looked at.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !loras
            .as_array()
            .expect("loras array")
            .iter()
            .any(|lora| lora["scope"] == "external"),
        "no external rows without an operator-configured root"
    );
}

/// sc-10452: the whole point of detecting a family on a scanned adapter is that the
/// existing compatibility gate then treats it exactly like a manifest LoRA. Nothing
/// declares `families` on an external row — `families_from_value_chain` falls back to
/// the singular `family` key and normalizes it — so assert the gate end-to-end rather
/// than trusting that fallback to survive a refactor.
#[test]
fn external_lora_family_is_detected_and_gates_model_compatibility() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let comfy_root = temp_dir.path().join("models");
    write_comfy_wan_adapter(
        &comfy_root
            .join("loras")
            .join("Wan")
            .join("detailz-wan.safetensors"),
    );

    let mut cache = crate::external_loras::ExternalLoraCache::default();
    let catalog =
        crate::external_loras::scan_external_loras(std::slice::from_ref(&comfy_root), &mut cache);
    let lora = catalog.first().expect("one scanned adapter");
    // Read from the tensor keys, not the filename or folder.
    assert_eq!(lora["family"], "wan-video");

    let lora_id = lora["id"].as_str().expect("id").to_owned();
    let attached = vec![json!({ "id": lora_id, "weight": 0.8 })];

    // A Wan model accepts it: the detected family matches `loraCompatibility.families`.
    let wan = vec![json!({
        "id": "wan_2_2",
        "loraCompatibility": { "families": ["wan-video"], "types": ["style"] }
    })];
    let accepted =
        crate::validate_lora_specs_for_model(&wan, &catalog, "wan_2_2", &attached, false, "LoRA")
            .expect("a wan-video adapter is compatible with a wan-video model");
    assert_eq!(accepted.len(), 1);

    // A Z-Image model rejects the same adapter, naming the detected family.
    let z_image = vec![json!({
        "id": "z_image_turbo",
        "loraCompatibility": { "families": ["z-image"], "types": ["style"] }
    })];
    let error = crate::validate_lora_specs_for_model(
        &z_image,
        &catalog,
        "z_image_turbo",
        &attached,
        false,
        "LoRA",
    )
    .expect_err("a wan-video adapter is not compatible with a z-image model");
    assert!(
        error.detail.contains("wan-video"),
        "the rejection names the detected family: {}",
        error.detail
    );
}

/// sc-10452: an adapter whose family the detector cannot identify (in the real tree:
/// an LLM LoRA, and the ComfyUI Qwen-Image adapters of sc-10506) is still LISTED —
/// the user pointed us at the folder and deserves to see we found the file — but the
/// API **fails closed** when one is attached to a generation. We will not guess a
/// family and load arbitrary tensors into a model.
#[test]
fn external_lora_without_a_detected_family_is_refused_at_job_create() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let comfy_root = temp_dir.path().join("models");
    let adapter = comfy_root.join("loras").join("mystery.safetensors");
    std::fs::create_dir_all(adapter.parent().expect("parent")).expect("mkdir");
    write_test_safetensors_with_keys(&adapter, &["some.unknown.tensor".to_owned()]);

    let mut cache = crate::external_loras::ExternalLoraCache::default();
    let catalog =
        crate::external_loras::scan_external_loras(std::slice::from_ref(&comfy_root), &mut cache);

    // Listed, installed, but carrying no family.
    let lora = catalog.first().expect("the adapter is still surfaced");
    assert_eq!(lora["installState"], "installed");
    assert!(lora.get("family").is_none(), "no family could be detected");

    let attached = vec![json!({ "id": lora["id"].as_str().expect("id"), "weight": 0.8 })];
    let models = vec![json!({
        "id": "wan_2_2",
        "loraCompatibility": { "families": ["wan-video"], "types": ["style"] }
    })];

    let error = crate::validate_lora_specs_for_model(
        &models, &catalog, "wan_2_2", &attached, false, "LoRA",
    )
    .expect_err("an unidentifiable adapter must not be loaded into a model");
    assert!(
        error.detail.contains("no declared family"),
        "the refusal explains why: {}",
        error.detail
    );
}
