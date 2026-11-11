use super::{
    model_import_request_from_multipart, validate_minimax_h3_import_declaration, ModelImportRequest,
};
use crate::tests::support::*;
use axum::extract::FromRequest;

fn app(temp: &tempfile::TempDir) -> axum::Router {
    let manifests = temp.path().join("config/manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    // H3 need not support LoRAs to be a supported model-import architecture.
    std::fs::write(manifests.join("builtin.models.jsonc"), json!({
            "schemaVersion": 1,
            "models": [
                { "id": "minimax_h3", "type": "video", "family": "minimax-h3", "loraCompatibility": { "families": [] },
                  "downloads":[
                    {"provider":"huggingface","repo":"fixture/tiers","revision":"1111111111111111111111111111111111111111","variant":"q4","default":true,"files":["q4/transformer/*"]},
                    {"provider":"huggingface","repo":"fixture/tiers","revision":"1111111111111111111111111111111111111111","variant":"bf16","files":["bf16/transformer/*"]},
                    {"provider":"huggingface","repo":"fixture/tiers","revision":"1111111111111111111111111111111111111111","variant":"q4","coRequisite":true,"files":["q4/transformer_ref/*"]},
                    {"provider":"huggingface","repo":"fixture/shared","revision":"1111111111111111111111111111111111111111","coRequisite":true,"componentId":"tokenizer","files":["tokenizer/config.json"]}
                  ] },
                { "id":"minimax_h3_ref", "type":"video", "family":"minimax-h3", "downloads":[
                    {"provider":"huggingface","repo":"fixture/tiers","revision":"1111111111111111111111111111111111111111","variant":"q4","default":true,"files":["q4/transformer_ref/*"]},
                    {"provider":"huggingface","repo":"fixture/tiers","revision":"1111111111111111111111111111111111111111","variant":"bf16","files":["bf16/transformer_ref/*"]},
                    {"provider":"huggingface","repo":"fixture/tiers","revision":"1111111111111111111111111111111111111111","variant":"q4","coRequisite":true,"files":["q4/transformer/*"]},
                    {"provider":"huggingface","repo":"fixture/shared","revision":"1111111111111111111111111111111111111111","coRequisite":true,"componentId":"tokenizer","files":["tokenizer/config.json"]}
                  ] },
                { "id": "sdxl", "type": "image", "family": "sdxl" }
            ]
        }).to_string()).unwrap();
    create_app(test_settings(temp)).unwrap()
}

/// Materialize one REGISTERED minimax_h3 component row in `snapshot` so the shared completeness
/// predicate (`minimax_h3_import::import_component_files_complete`) accepts it: explicit files as
/// written, and a wildcard component directory with the minimum loadable closure for its kind.
/// Returns the files written, the weights file last.
fn seed_registered_h3_component(
    snapshot: &std::path::Path,
    row: &Value,
) -> Vec<std::path::PathBuf> {
    let mut written = Vec::new();
    let mut write = |path: std::path::PathBuf, bytes: &[u8]| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        written.push(path);
    };
    let files = row["files"].as_array().unwrap();
    let shards = files
        .iter()
        .filter_map(|file| file.as_str()?.rsplit('/').next())
        .filter(|name| name.starts_with("model-") && name.ends_with(".safetensors"))
        .map(|name| (name.to_owned(), json!(name)))
        .collect::<serde_json::Map<_, _>>();
    for file in files {
        let file = file.as_str().unwrap();
        let Some(dir) = file.strip_suffix("/*") else {
            let bytes = if file.ends_with("model.safetensors.index.json") {
                json!({"weight_map": shards}).to_string().into_bytes()
            } else {
                b"{}".to_vec()
            };
            write(snapshot.join(file), &bytes);
            continue;
        };
        let dir = snapshot.join(dir);
        match row["componentId"].as_str().unwrap_or_default() {
            "text_encoder" => {
                write(dir.join("config.json"), b"{}");
                write(
                    dir.join("model.safetensors.index.json"),
                    br#"{"weight_map":{"w":"model.safetensors"}}"#,
                );
                write(dir.join("model.safetensors"), b"weights");
            }
            "tokenizer" => {
                write(dir.join("tokenizer_config.json"), b"{}");
                write(dir.join("tokenizer.json"), b"{}");
            }
            _ => {
                write(dir.join("config.json"), b"{}");
                write(dir.join("diffusion_pytorch_model.safetensors"), b"weights");
            }
        }
    }
    written
}

#[tokio::test]
async fn explicit_three_partitions_are_queued_with_persisted_catalog_fields() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let app = app(&temp);
    for (partition, base) in [
        ("fl2va", "minimax_h3"),
        ("ref2va", "minimax_h3_ref"),
        ("hybrid", "minimax_h3"),
    ] {
        let (status, job) = request(
            app.clone(),
            "POST",
            "/api/v1/models/import",
            json!({
                "name": format!("Custom H3 {partition}"),
                "source": { "kind": "huggingFace", "repo": "owner/h3-finetune" },
                "family": "minimax-h3", "baseModel": base, "minimaxH3Partition": partition,
                "licenseAcknowledged": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        assert_eq!(job["payload"]["baseModel"], base);
        assert_eq!(job["payload"]["minimaxH3Partition"], partition);
        let entry = &job["payload"]["manifestEntry"];
        assert_eq!(entry["baseModel"], base);
        assert_eq!(entry["minimaxH3Partition"], partition);
        assert_eq!(entry["type"], "video");
        let rows = entry["downloads"].as_array().unwrap();
        // The import inherits its base's shared component floor. Both bases are REGISTERED, so the
        // floor is the compiled registry row's (the dense q4 encoder, both VAEs, the tokenizer and
        // the FL2VA constructor documents), not the miniature config-dir rows above. Single-partition
        // imports also carry the opposite stock DiT as their companion.
        let mut components = rows
            .iter()
            .filter_map(|row| row["componentId"].as_str())
            .collect::<Vec<_>>();
        components.sort_unstable();
        components.dedup();
        for shared in [
            "audio_vae",
            "fl2va_audio_vae_config",
            "import_transformer_config",
            "text_encoder",
            "tokenizer",
            "video_vae",
        ] {
            assert!(
                components.contains(&shared),
                "{partition}: {shared}: {rows:#?}"
            );
        }
        assert_eq!(
            rows.len(),
            if partition == "hybrid" { 6 } else { 7 },
            "{rows:#?}"
        );
        let config = rows
            .iter()
            .find(|row| row["componentId"] == "import_transformer_config")
            .unwrap();
        // The config row is provisioned from the registered base's own pinned rehost.
        let base_pin = sceneworks_core::builtin_manifests::builtin_model_entry(base)
            .expect("registered base")["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["repo"] == "SceneWorks/minimax-h3-mlx")
            .expect("registered rehost row")["revision"]
            .clone();
        assert_eq!(config["revision"], base_pin);
        assert_eq!(
            config["files"],
            json!([if partition == "ref2va" {
                "bf16/transformer_ref/config.json"
            } else {
                "bf16/transformer/config.json"
            }])
        );
        assert!(rows.iter().all(super::is_co_requisite_download));
        assert!(
            rows.iter().all(|row| row["revision"]
                .as_str()
                .is_some_and(|revision| revision.len() == 40
                    && revision.bytes().all(|byte| byte.is_ascii_hexdigit()))),
            "E8/E9: every queued resident declaration must pin its source revision"
        );
        assert!(
            entry.get("importPlan").is_none(),
            "uninspected bytes are not ready"
        );
        let imported_slot = if partition == "ref2va" {
            "q4/transformer_ref/*"
        } else {
            "q4/transformer/*"
        };
        assert!(
            !rows
                .iter()
                .any(|row| super::string_array_field(row, "files")
                    .contains(&imported_slot.to_owned()))
        );
    }
    // Negative control: an absent component revision must refuse (E8/E9), never float to main.
    // The bases are REGISTERED and the compiled registry pins every row at build time, so a
    // config-dir edit can no longer author an unpinned resident for them; the refusal is exercised
    // on the seam every H3 import admission freezes its component declaration through.
    let mut unpinned = sceneworks_core::builtin_manifests::builtin_model_entry("minimax_h3")
        .expect("registered base")
        .clone();
    for row in unpinned["downloads"].as_array_mut().unwrap() {
        if row["componentId"] == "tokenizer" {
            row.as_object_mut().unwrap().remove("revision");
        }
    }
    let error =
        sceneworks_core::model_artifacts::provenance::frozen_h3_component_downloads(&unpinned)
            .expect_err("an unpinned resident declaration must refuse");
    assert!(error.contains("component revision is missing"), "{error}");
    drop(app);
}

#[tokio::test]
async fn missing_and_incompatible_declarations_fail_by_field_name() {
    let temp = tempfile::tempdir().unwrap();
    let app = app(&temp);
    for (declaration, field) in [
        (json!({}), "minimaxH3Partition"),
        (json!({ "baseModel": "minimax_h3" }), "minimaxH3Partition"),
        (
            json!({ "baseModel": "minimax_h3", "minimaxH3Partition": "auto" }),
            "minimaxH3Partition",
        ),
        (json!({ "minimaxH3Partition": "fl2va" }), "baseModel"),
        (
            json!({ "minimaxH3Partition": "fl2va", "baseModel": "minimax_h3_ref" }),
            "baseModel",
        ),
        (
            json!({ "minimaxH3Partition": "ref2va", "baseModel": "minimax_h3" }),
            "baseModel",
        ),
        (
            json!({ "minimaxH3Partition": "hybrid", "baseModel": "minimax_h3_ref" }),
            "baseModel",
        ),
    ] {
        let mut body =
            json!({ "name": "Custom H3", "repo": "owner/custom", "family": "minimax-h3" });
        body.as_object_mut()
            .unwrap()
            .extend(declaration.as_object().unwrap().clone());
        let (status, error) = request(app.clone(), "POST", "/api/v1/models/import", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(error["detail"].as_str().unwrap().contains(field), "{error}");
    }
}

#[test]
fn detected_h3_without_a_declaration_is_not_assigned_semantics() {
    let mut payload: ModelImportRequest =
        serde_json::from_value(json!({ "repo": "owner/custom" })).unwrap();
    validate_minimax_h3_import_declaration(&payload).unwrap();
    // Family reconciliation may discover H3 even when the request left family unspecified.
    payload.family = Some("minimax-h3".to_owned());
    let error = validate_minimax_h3_import_declaration(&payload).unwrap_err();
    assert!(error.detail.contains("minimaxH3Partition"));
    assert!(payload.base_model.is_none());
    assert!(payload.minimax_h3_partition.is_none());
    payload.family = Some("sdxl".to_owned());
    validate_minimax_h3_import_declaration(&payload).unwrap();
    payload.base_model = Some("minimax_h3".to_owned());
    payload.minimax_h3_partition = Some("hybrid".to_owned());
    assert!(validate_minimax_h3_import_declaration(&payload)
        .unwrap_err()
        .detail
        .contains("family"));
}

#[tokio::test]
async fn multipart_preserves_the_same_explicit_declaration_names() {
    let temp = tempfile::tempdir().unwrap();
    let (_, state) = create_app_with_state(test_settings(&temp)).unwrap();
    for (partition, base) in [
        ("fl2va", "minimax_h3"),
        ("ref2va", "minimax_h3_ref"),
        ("hybrid", "minimax_h3"),
    ] {
        let boundary = "H3_DECLARATION_BOUNDARY";
        let mut body = String::new();
        for (name, value) in [
            ("family", "minimax-h3"),
            ("baseModel", base),
            ("minimaxH3Partition", partition),
        ] {
            body.push_str(&format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"));
        }
        body.push_str(&format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"h3.safetensors\"\r\nContent-Type: application/octet-stream\r\n\r\nheader\r\n--{boundary}--\r\n"));
        let request = Request::builder()
            .method("POST")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = axum::extract::Multipart::from_request(request, &state)
            .await
            .unwrap();
        let (payload, path) = model_import_request_from_multipart(&state, multipart)
            .await
            .unwrap();
        assert_eq!(payload.base_model.as_deref(), Some(base));
        assert_eq!(payload.minimax_h3_partition.as_deref(), Some(partition));
        validate_minimax_h3_import_declaration(&payload).unwrap();
        super::cleanup_staged_model_upload(&path).await;
    }
}

#[tokio::test]
async fn linked_h3_imports_keep_explicit_partitions_and_reject_missing_or_incompatible_fields() {
    let temp = tempfile::tempdir().unwrap();
    let app = app(&temp);
    let library = temp.path().join("library");
    std::fs::create_dir_all(&library).unwrap();
    let (status, root) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({
            "path": library.to_str().unwrap(),
        }),
        "127.0.0.1:54321",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{root}");
    for (declaration, accepted, field) in [
        (
            json!({ "baseModel": "minimax_h3", "minimaxH3Partition": "fl2va" }),
            true,
            "",
        ),
        (
            json!({ "baseModel": "minimax_h3_ref", "minimaxH3Partition": "ref2va" }),
            true,
            "",
        ),
        (
            json!({ "baseModel": "minimax_h3", "minimaxH3Partition": "hybrid" }),
            true,
            "",
        ),
        (json!({}), false, "minimaxH3Partition"),
        (
            json!({ "minimaxH3Partition": "ref2va" }),
            false,
            "baseModel",
        ),
        (
            json!({ "baseModel": "minimax_h3", "minimaxH3Partition": "ref2va" }),
            false,
            "baseModel",
        ),
    ] {
        let mut body = json!({
            "linkedRootId": root["rootId"],
            "linkedRelativePath": "custom/transformer",
            "family": "minimax-h3",
            "licenseAcknowledged": true,
        });
        body.as_object_mut()
            .unwrap()
            .extend(declaration.as_object().unwrap().clone());
        let (status, response) = request(app.clone(), "POST", "/api/v1/models/import", body).await;
        if accepted {
            assert_eq!(status, StatusCode::CREATED, "{response}");
            let entry = &response["payload"]["manifestEntry"];
            assert_eq!(entry["baseModel"], declaration["baseModel"]);
            assert_eq!(
                entry["minimaxH3Partition"],
                declaration["minimaxH3Partition"]
            );
            assert_eq!(entry["type"], "video");
            assert_eq!(entry["source"]["rootId"], root["rootId"]);
            assert!(entry.get("paths").is_none());
            assert!(response["payload"].get("targetDir").is_none());
        } else {
            assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
            assert!(
                response["detail"].as_str().unwrap().contains(field),
                "{response}"
            );
        }
    }
}
#[tokio::test]
async fn omitted_family_h3_declarations_constrain_remote_and_linked_worker_jobs() {
    let temp = tempfile::tempdir().unwrap();
    let app = app(&temp);
    let library = temp.path().join("library");
    std::fs::create_dir_all(&library).unwrap();
    let (status, root) = request_with_peer(
        app.clone(),
        "POST",
        "/api/v1/models/library-roots",
        json!({"path":library.to_str().unwrap()}),
        "127.0.0.1:54321",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{root}");
    for source in [
        json!({"name":"Deferred H3 remote","source":{"kind":"huggingFace","repo":"owner/uninspected-weights"}}),
        json!({"name":"Deferred H3 linked","linkedRootId":root["rootId"],"linkedRelativePath":"uninspected.safetensors"}),
    ] {
        let mut body = source;
        body["baseModel"] = json!("minimax_h3");
        body["minimaxH3Partition"] = json!("fl2va");
        body["licenseAcknowledged"] = json!(true);
        let (status, response) = request(app.clone(), "POST", "/api/v1/models/import", body).await;
        assert_eq!(status, StatusCode::CREATED, "{response}");
        assert_eq!(response["payload"]["family"], "minimax-h3");
        assert_eq!(response["payload"]["manifestEntry"]["family"], "minimax-h3");
        assert!(
            response["payload"]["manifestEntry"]
                .get("importPlan")
                .is_none(),
            "uninspected bytes are never registered ready by the API"
        );
    }
}

#[tokio::test]
async fn local_sdxl_bytes_refuse_h3_declarations_even_when_family_is_omitted() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let source = settings
        .data_dir
        .join("models/incoming/foreign.safetensors");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    let mut header = serde_json::Map::new();
    for (index, key) in [
        "model.diffusion_model.input_blocks.7.0.out_layers.3.weight",
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        "conditioner.embedders.1.model.transformer.resblocks.9.attn.in_proj_weight",
        "first_stage_model.encoder.conv_in.weight",
        "first_stage_model.decoder.conv_out.weight",
    ]
    .iter()
    .enumerate()
    {
        header.insert(
            (*key).to_owned(),
            json!({"dtype":"F16","shape":[1],"data_offsets":[index*2,index*2+2]}),
        );
    }
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.extend([0_u8; 12]);
    std::fs::write(&source, &bytes).unwrap();
    assert_eq!(
        super::import_source_supported(&source).unwrap().as_deref(),
        Some("sdxl")
    );
    let app = app(&temp);
    let (status, response) = request(app.clone(), "POST", "/api/v1/models/import", json!({"name":"Wrong H3 architecture","sourcePath":source.to_str().unwrap(),"baseModel":"minimax_h3","minimaxH3Partition":"fl2va"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let detail = response["detail"].as_str().unwrap();
    assert!(
        detail.contains("source family") && detail.contains("sdxl"),
        "{response}"
    );
    assert!(!settings
        .config_dir
        .join("manifests/user.models.jsonc")
        .exists());
    assert_eq!(std::fs::read(&source).unwrap(), bytes);
    // The exact same supported source remains a normal image import without H3 compatibility.
    let (status, response) = request(
        app,
        "POST",
        "/api/v1/models/import",
        json!({"name":"Normal SDXL architecture","sourcePath":source.to_str().unwrap()}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{response}");
    assert_eq!(response["payload"]["manifestEntry"]["family"], "sdxl");
    assert_eq!(response["payload"]["manifestEntry"]["type"], "image");
}

fn write_lifecycle_h3(dir: &std::path::Path) {
    let mut config = sceneworks_core::minimax_h3_codec::H3Config::published().0;
    for (key, value) in [
        ("num_layers", 1),
        ("num_refiner_layers", 1),
        ("num_attention_heads", 2),
        ("attention_head_dim", 6),
        ("hidden_size", 8),
        ("ffn_dim", 8),
        ("in_channels", 2),
        ("audio_in_channels", 2),
        ("text_dim", 8),
        ("freq_dim", 4),
        ("time_embed_hidden_dim", 8),
        ("time_embed_dim", 4),
        ("rope_freq_dim", 1),
    ] {
        config[key] = json!(value);
    }
    let shapes = sceneworks_core::minimax_h3_codec::H3Config(config.clone())
        .shapes()
        .unwrap();
    let mut header = serde_json::Map::new();
    let mut offset = 0;
    for (name, shape) in shapes {
        let size = shape.iter().product::<usize>() * 2;
        header.insert(
            name,
            json!({"dtype":"BF16","shape":shape,"data_offsets":[offset,offset+size]}),
        );
        offset += size;
    }
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset, 0x31);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), bytes).unwrap();
}

#[tokio::test]
async fn h3_card_repairs_only_the_missing_bound_component_not_the_imported_dit() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let _ = app(&temp);
    let settings = test_settings(&temp);
    let library = temp.path().join("library");
    let source = library.join("custom");
    write_lifecycle_h3(&source);
    let _cache_root = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
        settings.data_dir.clone(),
        temp.path().join("fixture-hub"),
    );
    let store =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
    let root = store.approve_root(&library).unwrap();
    // The selected base is REGISTERED: it authorizes exactly its compiled rows, so the import binds
    // the registered audio VAE and q4 packed encoder rows (a config-dir edit of the base can no
    // longer add resident rows for it). Each wildcard selector is materialized as two files.
    let base = sceneworks_core::builtin_manifests::builtin_model_entry("minimax_h3")
        .expect("registered base");
    let host = if cfg!(target_os = "macos") {
        "macos"
    } else {
        std::env::consts::OS
    };
    let registered_row = |component: &str, variant: Option<&str>| {
        base["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| {
                row["componentId"] == component
                    && variant.is_none_or(|variant| row["variant"] == variant)
                    && row["platforms"]
                        .as_array()
                        .is_none_or(|platforms| platforms.contains(&json!(host)))
            })
            .unwrap_or_else(|| panic!("registered {component} row"))
            .clone()
    };
    let rows = json!([
        registered_row("audio_vae", None),
        registered_row("text_encoder", Some("q4")),
    ]);
    let mut audio_weights = None;
    let mut entry = json!({"id":"repair_h3","name":"Repair H3", "family":"minimax-h3", "baseModel":"minimax_h3","minimaxH3Partition":"fl2va","importComponentTier":"q4", "source":{"provider":"linked-library","rootId":root.root_id,"relativePath":"custom"}, "downloads":rows}).as_object().unwrap().clone();
    for row in rows.as_array().unwrap() {
        let cache =
            super::huggingface_repo_cache_path(&settings.data_dir, row["repo"].as_str().unwrap())
                .unwrap();
        assert!(cache.starts_with(temp.path()));
        let written = seed_registered_h3_component(
            &cache
                .join("snapshots")
                .join(row["revision"].as_str().unwrap()),
            row,
        );
        if row["componentId"] == "audio_vae" {
            audio_weights = written.last().cloned();
        }
    }
    // E8/E9: capture complete contents before compile, then remove only the repair target.
    let context = sceneworks_worker::minimax_h3_import_fixture::captured_component_context(
        &settings.data_dir,
        &entry,
    );
    let compiled = store
        .compile_linked_with_context(&root.root_id, "custom", context)
        .unwrap();
    entry.insert(
        "importPlan".into(),
        json!({"checkpointId":compiled.checkpoint_id,
            "planId":compiled.plan.plan_id,"semanticDigest":compiled.record.plan.semantic_digest}),
    );
    let assembly = sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(
        &compiled.plan,
        &entry,
    )
    .unwrap();
    sceneworks_core::minimax_h3_import::stamp_imported_minimax_h3_entry(&mut entry, &assembly);
    assert_h3_component_authority_required(&settings.data_dir, &entry);
    let manifest = settings.config_dir.join("manifests/user.models.jsonc");
    std::fs::write(
        &manifest,
        json!({"schemaVersion":1,"models":[entry]}).to_string(),
    )
    .unwrap();
    // Start complete; remove ONLY audio VAE bytes. The import and text encoder remain sound.
    let audio_row = &rows[0];
    let audio = audio_weights.expect("the audio VAE weights were seeded");
    std::fs::remove_file(&audio).unwrap();
    let router = create_app(settings.clone()).unwrap();
    let (status, catalog) = request(router.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{catalog}");
    let projected = catalog
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == "repair_h3")
        .unwrap();
    assert_eq!(projected["downloadable"], false);
    assert_eq!(projected["componentReadiness"]["ready"], false);
    let components = projected["componentReadiness"]["components"]
        .as_array()
        .unwrap();
    assert_eq!(
        components
            .iter()
            .find(|row| row["source"] == "plan")
            .unwrap()["present"],
        true,
        "missing audio VAE must not mark the imported DiT absent"
    );
    let repairs = projected["repairActions"].as_array().unwrap();
    assert_eq!(repairs.len(), 1, "only audio VAE needs repair: {repairs:?}");
    assert_eq!(repairs[0]["componentId"], "audio_vae");
    let (status, error) = request(
        router.clone(),
        "POST",
        "/api/v1/models/repair_h3/download",
        json!({"componentId":repairs[0]["componentId"]}),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{error}");
    assert_eq!(error["code"], "license_acknowledgment_required");
    // F-209: accepting the card persists the displayed inherited terms too.
    let (status, accepted) = request(
        router.clone(),
        "PUT",
        "/api/v1/models/repair_h3/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    assert_eq!(accepted["acknowledged"], true);
    let (status, job) = request(
        router.clone(),
        "POST",
        "/api/v1/models/repair_h3/download",
        json!({"componentId":repairs[0]["componentId"]}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["repo"], audio_row["repo"]);
    assert_eq!(job["payload"]["revision"], audio_row["revision"]);
    assert_eq!(job["payload"]["files"], rows[0]["files"]);
    assert_eq!(job["payload"]["importComponentTier"], "q4");
    assert_eq!(job["payload"]["componentId"], "audio_vae");
    assert!(job["payload"].get("family").is_none());
    assert!(job["payload"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .all(|file| !file.as_str().unwrap().contains("transformer")));
    // Withdrawing a displayed inherited obligation invalidates hydration too.
    let inherited = sceneworks_core::minimax_h3_import::inherited_component_entries(
        projected.as_object().unwrap(),
    );
    let inherited_id = inherited
        .iter()
        .find(|row| row["requiresLicenseAcknowledgment"] == true || row["gated"] == true)
        .unwrap()["id"]
        .as_str()
        .unwrap();
    assert!(
        sceneworks_core::license_acknowledgments::revoke_acknowledgment(
            &sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir),
            inherited_id
        )
        .unwrap()
    );
    let (_, view) = request(
        router.clone(),
        "GET",
        "/api/v1/models/repair_h3/license-acknowledgment",
        Value::Null,
    )
    .await;
    assert_eq!(view["acknowledged"], false, "{view}");
    let (status, error) = request(
        router.clone(),
        "POST",
        "/api/v1/models/repair_h3/download",
        json!({"componentId":"audio_vae"}),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{error}");
    request(
        router.clone(),
        "PUT",
        "/api/v1/models/repair_h3/license-acknowledgment",
        Value::Null,
    )
    .await;
    for body in [
        json!({}),
        json!({"componentId":"transformer"}),
        json!({"componentId":"audio_vae","variant":"q8"}),
        json!({"componentId":"text_encoder"}),
        json!({"componentId":""}),
    ] {
        let (status, response) = request(
            router.clone(),
            "POST",
            "/api/v1/models/repair_h3/download",
            body.clone(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {response}");
    }
    let (status, jobs) = request(router.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{jobs}");
    assert_eq!(
        jobs.as_array().unwrap().len(),
        1,
        "repair must not queue a stock DiT or unrelated component: {jobs}"
    );
    assert!(store.resolve(&compiled.checkpoint_id).is_ok());
    let repair_job_id = job["id"].as_str().unwrap();
    for operation in ["retry", "duplicate"] {
        let (status, replay) = request(
            router.clone(),
            "POST",
            &format!("/api/v1/jobs/{repair_job_id}/{operation}"),
            json!({"payloadChanges":{}}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {replay}");
        let (status, replay) = request(
            router.clone(),
            "POST",
            &format!("/api/v1/jobs/{repair_job_id}/{operation}"),
            json!({"payloadChanges":{"componentId":"text_encoder"}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {replay}");
    }
    assert!(
        sceneworks_core::license_acknowledgments::revoke_acknowledgment(
            &sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir),
            inherited_id
        )
        .unwrap()
    );
    for operation in ["retry", "duplicate"] {
        let (status, replay) = request(
            router.clone(),
            "POST",
            &format!("/api/v1/jobs/{repair_job_id}/{operation}"),
            json!({"payloadChanges":{}}),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{operation}: {replay}");
    }
}

#[tokio::test]
async fn h3_public_rescan_readiness_and_removal_keep_component_and_source_ownership() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let _ = app(&temp);
    let settings = test_settings(&temp);
    let library = temp.path().join("library");
    let source = library.join("custom");
    write_lifecycle_h3(&source);
    let _cache_root = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
        settings.data_dir.clone(),
        temp.path().join("fixture-hub"),
    );
    let store =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
    let root = store.approve_root(&library).unwrap();
    let required = [
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "vae/config.json",
        "vae/weights.safetensors",
        "audio_vae/config.json",
        "audio_vae/weights.safetensors",
        "FL2VA/audio_vae/config.yaml",
    ];
    let revision = "0123456789012345678901234567890123456789";
    let mut entry = json!({"id":"lifecycle_h3","family":"minimax-h3","baseModel":"minimax_h3","minimaxH3Partition":"fl2va","importComponentTier":"q4", "source":{"provider":"linked-library","rootId":root.root_id,"relativePath":"custom"}, "downloads":[{"provider":"huggingface","repo":"fixture/shared","revision":revision,"coRequisite":true,"files":required}]}).as_object().unwrap().clone();
    let cache = super::huggingface_repo_cache_path(&settings.data_dir, "fixture/shared").unwrap();
    assert!(cache.starts_with(temp.path()));
    let snapshot = cache.join("snapshots").join(revision);
    for file in required {
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"fixture").unwrap();
    }
    // E7/E8/E9: fresh source and component tier/content are committed together, not rebound
    // after publication. Later disappearance is missing content, not historical unknown authority.
    let context = sceneworks_worker::minimax_h3_import_fixture::captured_component_context(
        &settings.data_dir,
        &entry,
    );
    let compiled = store
        .compile_linked_with_context(&root.root_id, "custom", context)
        .unwrap();
    entry.insert(
        "importPlan".into(),
        json!({"checkpointId":compiled.checkpoint_id,
            "planId":compiled.plan.plan_id,"semanticDigest":compiled.record.plan.semantic_digest}),
    );
    let assembly = sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(
        &compiled.plan,
        &entry,
    )
    .unwrap();
    sceneworks_core::minimax_h3_import::stamp_imported_minimax_h3_entry(&mut entry, &assembly);
    assert_h3_component_authority_required(&settings.data_dir, &entry);
    let mut model = Value::Object(entry);
    std::fs::remove_file(snapshot.join(required[0])).unwrap();
    let missing = super::install_state_for(None, &model, &settings.data_dir);
    assert!(!missing.installed);
    assert!(missing.cache_incomplete);
    std::fs::write(snapshot.join(required[0]), b"fixture").unwrap();
    assert!(super::install_state_for(None, &model, &settings.data_dir).installed);
    let mut unbound = model.clone();
    unbound["importPlan"]
        .as_object_mut()
        .unwrap()
        .remove("assembly");
    assert!(!super::install_state_for(None, &unbound, &settings.data_dir).installed);
    let mut stale = model.clone();
    stale["importPlan"]["planId"] = json!("checkpoint-plan-stale");
    assert!(!super::install_state_for(None, &stale, &settings.data_dir).installed);
    // Every component participates in readiness; a complete unrelated snapshot cannot mask a
    // missing file in the bound revision. Restoring it recovers without re-importing the DiT.
    for file in required {
        std::fs::remove_file(snapshot.join(file)).unwrap();
        let sibling = cache.join("snapshots/other").join(file);
        std::fs::create_dir_all(sibling.parent().unwrap()).unwrap();
        std::fs::write(sibling, b"wrong revision").unwrap();
        let state = super::install_state_for(None, &model, &settings.data_dir);
        assert!(!state.installed, "{file}");
        assert!(state
            .missing_required_files
            .iter()
            .any(|missing| missing.contains(file)));
        std::fs::write(snapshot.join(file), b"fixture").unwrap();
    }
    let manifest = settings.config_dir.join("manifests/user.models.jsonc");
    let mut alias = model.clone();
    alias["id"] = json!("lifecycle_h3_alias");
    std::fs::write(
        &manifest,
        json!({"schemaVersion":1,"models":[model, alias]}).to_string(),
    )
    .unwrap();
    let app = create_app(settings.clone()).unwrap();
    let lease = store.lease_source(&compiled.checkpoint_id).unwrap();
    let (status, response) = request(
        app.clone(),
        "DELETE",
        "/api/v1/models/lifecycle_h3?permanent=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{response}");
    assert_eq!(response["code"], "checkpoint_source_in_use");
    assert!(store.resolve(&compiled.checkpoint_id).is_ok());
    drop(lease);
    let weight = source.join("diffusion_pytorch_model.safetensors");
    let mut changed = std::fs::read(&weight).unwrap();
    *changed.last_mut().unwrap() ^= 1;
    std::fs::write(&weight, &changed).unwrap();
    assert!(!super::install_state_for(None, &model, &settings.data_dir).installed);
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/models/library-roots/{}/rescan", root.root_id),
        json!({"relativePath":"custom"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let saved: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(&manifest).unwrap(),
    ))
    .unwrap();
    model = saved["models"][0].clone();
    assert_ne!(model["importPlan"]["planId"], compiled.plan.plan_id);
    assert!(super::install_state_for(None, &model, &settings.data_dir).installed);
    let (status, response) = request(
        app.clone(),
        "DELETE",
        "/api/v1/models/lifecycle_h3?permanent=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert!(
        store.resolve(&compiled.checkpoint_id).is_ok(),
        "another catalog alias still owns this plan"
    );
    let (status, response) = request(
        app,
        "DELETE",
        "/api/v1/models/lifecycle_h3_alias?permanent=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(
        std::fs::read(&weight).unwrap(),
        changed,
        "linked removal never owns original bytes"
    );
    assert!(store.record(&compiled.checkpoint_id).is_err());
}

#[tokio::test]
async fn custom_h3_import_requires_the_selected_base_component_license_acknowledgment() {
    let temp = tempfile::tempdir().unwrap();
    let _ = app(&temp);
    let path = temp.path().join("config/manifests/builtin.models.jsonc");
    let mut catalog: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    catalog["models"][0]["requiresLicenseAcknowledgment"] = json!(true);
    catalog["models"][0]["licenseNotice"] = json!("Fixture restricted components");
    std::fs::write(path, catalog.to_string()).unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    let mut body = json!({"modelId":"licensed_h3","source":{"kind":"huggingFace","repo":"community/custom"},"baseModel":"minimax_h3","minimaxH3Partition":"fl2va"});
    let (status, error) = request(app.clone(), "POST", "/api/v1/models/import", body.clone()).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{error}");
    body["licenseAcknowledged"] = json!(true);
    let (status, job) = request(app, "POST", "/api/v1/models/import", body).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["manifestEntry"]["baseModel"], "minimax_h3");
}

#[tokio::test]
async fn import_persists_a_complete_bf16_only_shared_closure_as_its_generation_default() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let _ = app(&temp);
    let settings = test_settings(&temp);
    // minimax_h3 is registered, so its shared closure is the compiled registry row's, not a
    // config-dir miniature. Seed exactly that row's bf16 dense encoder (this host's platform row)
    // and nothing for q4, so bf16 is the only complete shared closure.
    let base = sceneworks_core::builtin_manifests::builtin_model_entry("minimax_h3")
        .expect("registered base");
    let encoder = base["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| {
            row["componentId"] == "text_encoder"
                && row["variant"] == "bf16"
                && row["platforms"].as_array().is_some_and(|platforms| {
                    platforms.contains(&json!(if cfg!(target_os = "macos") {
                        "macos"
                    } else {
                        std::env::consts::OS
                    }))
                })
        })
        .expect("the registered bf16 encoder row for this host")
        .clone();
    let revision = encoder["revision"].as_str().unwrap().to_owned();
    // Seed exactly the bf16 import closure the server computes from the registered base (shared
    // floor, the stock companion DiT and the transformer config), and nothing tier-specific for
    // q4/q8, so bf16 is the only complete resident closure.
    let declaration = json!({"baseModel":"minimax_h3","minimaxH3Partition":"fl2va"});
    let closure = sceneworks_core::minimax_h3_import::import_component_downloads(
        declaration.as_object().unwrap(),
        &Value::Object(base.clone()),
        Some("bf16"),
        std::env::consts::OS,
    )
    .expect("the registered base has a bf16 import closure");
    for row in &closure {
        let snapshot =
            super::huggingface_repo_cache_path(&settings.data_dir, row["repo"].as_str().unwrap())
                .unwrap()
                .join("snapshots")
                .join(row["revision"].as_str().unwrap());
        let partition = row["files"].as_array().unwrap().iter().find_map(|file| {
            let file = file.as_str()?;
            ["transformer_ref", "transformer"]
                .into_iter()
                .find(|partition| file == format!("bf16/{partition}/*"))
        });
        match partition {
            Some(partition) => crate::tests::catalog::seed_minimax_partition(
                &snapshot,
                "bf16",
                partition,
                &crate::tests::catalog::MINIMAX_SHARDS,
            ),
            None => {
                seed_registered_h3_component(&snapshot, row);
            }
        }
    }
    let incomplete = closure
        .iter()
        .filter(|row| {
            super::install_state::imported_component_presence(&settings.data_dir, row)
                != sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence::Installed
        })
        .collect::<Vec<_>>();
    assert!(
        incomplete.is_empty(),
        "the seeded bf16 closure is resident: {incomplete:#?}"
    );
    let app = create_app(settings).unwrap();
    let (status, job) = request(app,"POST","/api/v1/models/import",json!({"modelId":"bf16_only_import","source":{"kind":"huggingFace","repo":"community/custom"},"baseModel":"minimax_h3","minimaxH3Partition":"fl2va","licenseAcknowledged":true})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["manifestEntry"]["importComponentTier"],
        "bf16"
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["downloads"][0]["variant"],
        "bf16"
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["downloads"][0]["revision"],
        revision
    );
    let mut surface = job["payload"]["manifestEntry"].as_object().unwrap().clone();
    super::apply_imported_video_support_surface(&mut surface);
    assert_eq!(
        surface["runtimeQuantTiers"],
        json!(["bf16"]),
        "normal picker defaults can only select the bound closure"
    );
}

#[test]
fn config_only_provisioning_never_marks_the_stock_dit_installed() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let revision = "1111111111111111111111111111111111111111";
    let base = json!({"id":"minimax_h3","type":"video","family":"minimax-h3","downloads":[
        {"provider":"huggingface","repo":"fixture/base","revision":revision,"variant":"bf16","default":true,"files":["bf16/transformer/*"]}
    ]});
    let row = sceneworks_core::minimax_h3_import::import_config_download(
        &base,
        "q4",
        std::env::consts::OS,
    )
    .unwrap();
    assert_eq!(row["files"], json!(["bf16/transformer/config.json"]));
    assert_eq!(row["coRequisite"], true);
    let snapshot = super::huggingface_repo_cache_path(temp.path(), "fixture/base")
        .unwrap()
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(snapshot.join("bf16/transformer")).unwrap();
    std::fs::write(snapshot.join("bf16/transformer/config.json"), b"{}").unwrap();
    assert!(
        !super::install_state_for(
            super::model_download_context(&base).unwrap(),
            &base,
            temp.path()
        )
        .installed
    );
}
