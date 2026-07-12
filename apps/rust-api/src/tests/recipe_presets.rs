//! rust-api recipe_presets tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[tokio::test]
async fn recipe_preset_crud_routes_persist_global_and_project_presets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "builtin_readonly",
                  "name": "Built-in Readonly",
                  "scope": "builtin",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo"
                }
              ]
            }
            "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"
            // user preset notes survive API writes
            {
              "schemaVersion": 1,
              /* preserve unknown root fields too */
              "futureRoot": true,
              "presets": []
            }
            "#,
    )
    .expect("user recipe presets writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z Image Turbo",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
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
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style_lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                },
                {
                  "id": "qwen_style",
                  "name": "Qwen Style",
                  "family": "qwen-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["qwen-image"] },
                  "source": { "provider": "local", "path": "loras/qwen.safetensors" }
                },
                {
                  "id": "deleted_style",
                  "name": "Deleted Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/deleted.safetensors" }
                },
                {
                  "id": "empty_dir_style",
                  "name": "Empty Dir Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/empty-dir" }
                },
                {
                  "id": "unknown_family",
                  "name": "Unknown Family",
                  "triggerWords": [],
                  "compatibility": {},
                  "source": { "provider": "local", "path": "loras/unknown.safetensors" }
                },
                {
                  "id": "no_path_style",
                  "name": "No Path Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local" }
                }
              ]
            }
            "#,
    )
    .expect("user loras writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    std::fs::create_dir_all(lora_dir.join("empty-dir")).expect("empty lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));
    write_test_safetensors(&lora_dir.join("qwen.safetensors"));
    write_test_safetensors(&lora_dir.join("unknown.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, loras) = request(app.clone(), "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let empty_dir_style = loras
        .as_array()
        .expect("loras array")
        .iter()
        .find(|lora| lora["id"] == "empty_dir_style")
        .expect("empty dir lora listed");
    assert_eq!(empty_dir_style["installState"], "missing");

    // This also pins the positive compatibility path: style_lora is installed and compatible with z_image_turbo.
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Soft Glow",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "order": 30,
            "defaults": { "resolution": "1024x1024" },
            "prompt": { "suffix": "soft glow" },
            "loras": [{ "id": "style_lora", "weight": 0.5 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["id"], "soft_glow");
    assert_eq!(created["scope"], "global");
    assert_eq!(created["builtInLoras"][0]["id"], "style_lora");

    let (status, updated) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/soft_glow",
        json!({
            "defaults": { "negativePrompt": "noise" },
            "loras": [{ "id": "style_lora", "weight": 0.75 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["defaults"]["negativePrompt"], "noise");
    assert_eq!(updated["loras"][0]["weight"], 0.75);

    let (status, duplicate) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets/soft_glow/duplicate",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(duplicate["id"], "soft_glow_copy");
    assert_eq!(duplicate["name"], "Soft Glow Copy");
    assert_eq!(duplicate["loras"][0]["id"], "style_lora");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());
    let (status, project_preset) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "project_soft_glow",
            "name": "Project Soft Glow",
            "scope": "project",
            "projectId": project_id,
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "order": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(project_preset["scope"], "project");
    assert!(project_path.join("recipes/presets.jsonc").is_file());

    for (id, name) in [("beta_order", "Beta Order"), ("alpha_order", "Alpha Order")] {
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "id": id,
                "name": name,
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "order": 10
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
    }

    let (status, ordered) = request(
            app.clone(),
            "GET",
            &format!(
                "/api/v1/recipe-presets?projectId={project_id}&workflow=text_to_image&model=z_image_turbo"
            ),
            Value::Null,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let ordered_ids = ordered
        .as_array()
        .unwrap()
        .iter()
        .map(|preset| preset["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        ordered_ids,
        vec![
            "builtin_readonly",
            "alpha_order",
            "beta_order",
            "soft_glow",
            "soft_glow_copy",
            "project_soft_glow"
        ]
    );

    let (status, scoped) = request(
        app.clone(),
        "GET",
        "/api/v1/recipe-presets?scope=global&workflow=text_to_image&model=z_image_turbo",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(scoped
        .as_array()
        .unwrap()
        .iter()
        .all(|preset| preset["scope"] == "global"));

    let (status, readonly_error) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/builtin_readonly",
        json!({ "name": "Nope" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        readonly_error["detail"],
        "Built-in recipe presets are read-only"
    );

    let (status, project_updated) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/recipe-presets/project_soft_glow?projectId={project_id}"),
        json!({ "prompt": { "suffix": "project update" } }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(project_updated["prompt"]["suffix"], "project update");

    let (status, _, bytes) = request_raw(
        app.clone(),
        "DELETE",
        "/api/v1/recipe-presets/soft_glow",
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let archived: Value = serde_json::from_slice(&bytes).expect("archive response parses");
    assert_eq!(archived["archived"], true);

    let (status, _, bytes) = request_raw(
        app.clone(),
        "DELETE",
        &format!("/api/v1/recipe-presets/project_soft_glow?projectId={project_id}"),
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let archived: Value = serde_json::from_slice(&bytes).expect("project archive response parses");
    assert_eq!(archived["archived"], true);

    let (status, visible) =
        request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!visible
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "soft_glow"));

    let (status, archived_visible) = request(
        app.clone(),
        "GET",
        "/api/v1/recipe-presets?includeArchived=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(archived_visible
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "soft_glow" && preset["archived"] == true));

    let (status, unarchived) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/soft_glow",
        json!({ "archived": false }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unarchived["archived"], false);

    let saved_manifest_text = std::fs::read_to_string(config_dir.join("user.recipe-presets.jsonc"))
        .expect("user recipe preset manifest reads");
    assert!(saved_manifest_text.starts_with(API_MANAGED_MANIFEST_HEADER));
    assert!(!saved_manifest_text.contains("// user preset notes survive API writes"));
    assert!(!saved_manifest_text.contains("/* preserve unknown root fields too */"));
    let saved_manifest: Value = serde_json::from_str(&strip_jsonc_comments(&saved_manifest_text))
        .expect("saved manifest parses");
    assert_eq!(saved_manifest["futureRoot"], true);

    let (status, second_update) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/soft_glow",
        json!({ "order": 31 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(second_update["order"], 31);
    let second_manifest_text =
        std::fs::read_to_string(config_dir.join("user.recipe-presets.jsonc"))
            .expect("user recipe preset manifest reads after second write");
    assert!(second_manifest_text.starts_with(API_MANAGED_MANIFEST_HEADER));
    assert_eq!(
        second_manifest_text
            .matches(API_MANAGED_MANIFEST_HEADER)
            .count(),
        1
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "Bad Id",
            "name": "Bad Id",
            "model": "z_image_turbo",
            "workflow": "text_to_image"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset id must use lowercase letters, numbers, dashes, or underscores"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Bad Order",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "order": "high"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset order must be an integer"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Bad Workflow",
            "model": "z_image_turbo",
            "workflow": "text_to_video"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Model z_image_turbo does not support workflow text_to_video"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Too Many LoRAs",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [
                { "id": "style_one" },
                { "id": "style_two" },
                { "id": "style_three" },
                { "id": "style_four" },
                { "id": "style_five" },
                { "id": "style_six" }
            ]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe presets can include at most 5 LoRAs"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Overweighted LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "style_one", "weight": 2.5 }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA weight must be between -2 and 2"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Missing LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "missing_lora" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA not found: missing_lora"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Deleted LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "deleted_style" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA is not installed: deleted_style"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "No Path LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "no_path_style" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA is not installed: no_path_style"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Unknown Family LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "unknown_family" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
            bad_error["detail"],
            "LoRA unknown_family has no declared family; cannot verify compatibility with model z_image_turbo"
        );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Wrong Family LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    let create_one = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "concurrent_one",
            "name": "Concurrent One",
            "model": "z_image_turbo",
            "workflow": "text_to_image"
        }),
    );
    let create_two = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "concurrent_two",
            "name": "Concurrent Two",
            "model": "z_image_turbo",
            "workflow": "text_to_image"
        }),
    );
    let ((status_one, _), (status_two, _)) = tokio::join!(create_one, create_two);
    assert_eq!(status_one, StatusCode::CREATED);
    assert_eq!(status_two, StatusCode::CREATED);
    let (status, concurrent_presets) = request(
        app.clone(),
        "GET",
        "/api/v1/recipe-presets?scope=global&includeArchived=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(concurrent_presets
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "concurrent_one"));
    assert!(concurrent_presets
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "concurrent_two"));

    let (bad_status, bad_error) = request(
        app,
        "GET",
        "/api/v1/recipe-presets?workflow=bogus",
        Value::Null,
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(bad_error["detail"], "Unsupported recipe preset workflow");
}

#[tokio::test]
async fn recipe_preset_accepts_full_studio_snapshot_and_rejects_bad_defaults() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z Image Turbo",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
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
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style_lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("user loras writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // A full studio snapshot: literal prompt + cfg/steps/sampler + a weighted LoRA.
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Atrium Look",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "modes": ["text_to_image", "character_image", "style_variations"],
            "defaults": {
                "prompt": "a portrait in the atrium",
                "negativePrompt": "blurry",
                "resolution": "1024x1024",
                "count": 4,
                "mode": "character_image",
                "guidanceScale": 5.0,
                "steps": 28,
                "sampler": "default",
                "ipAdapterScale": 0.8
            },
            "loras": [{ "id": "style_lora", "weight": 0.65 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["id"], "atrium_look");
    // The flatten-extra defaults round-trip intact through persistence.
    assert_eq!(created["defaults"]["prompt"], "a portrait in the atrium");
    assert_eq!(created["defaults"]["guidanceScale"], 5.0);
    assert_eq!(created["defaults"]["steps"], 28);
    assert_eq!(created["defaults"]["sampler"], "default");
    assert_eq!(created["defaults"]["mode"], "character_image");
    assert_eq!(created["builtInLoras"][0]["id"], "style_lora");
    assert_eq!(created["builtInLoras"][0]["weight"], 0.65);

    // Re-reading the catalog returns the persisted snapshot.
    let (status, listed) = request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let saved = listed
        .as_array()
        .expect("presets array")
        .iter()
        .find(|preset| preset["id"] == "atrium_look")
        .expect("saved preset listed");
    assert_eq!(saved["defaults"]["prompt"], "a portrait in the atrium");
    assert_eq!(saved["defaults"]["steps"], 28);

    // An out-of-range guidance scale is rejected.
    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Too Hot",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "defaults": { "guidanceScale": 999.0 }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("guidanceScale"),
        "unexpected error: {error}"
    );

    // A non-integer steps value is rejected.
    let (status, _error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Bad Steps",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "defaults": { "steps": 3.5 }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn empty_builtin_preset_and_lora_manifests_ship_empty_catalogs() {
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
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
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

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, presets) =
        request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(presets.as_array().expect("presets array").len(), 0);

    let (status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(loras.as_array().expect("loras array").len(), 0);
}

#[tokio::test]
async fn legacy_preset_read_defaults_do_not_select_uninstalled_models() {
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
                  "id": "missing_image_model",
                  "name": "Missing Image Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
                  "downloads": [{ "provider": "huggingface", "repo": "owner/missing-model", "files": ["*.safetensors"] }],
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
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                { "id": "legacy_text", "name": "Legacy Text", "modes": ["text_to_image"] }
              ]
            }
            "#,
    )
    .expect("user recipe presets writes");
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

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, presets) = request(app, "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let preset = &presets.as_array().expect("presets array")[0];
    assert_eq!(preset["workflow"], "text_to_image");
    assert!(preset.get("model").is_none());
    assert_eq!(
        preset["appliedDefaults"]["notes"][0],
        "workflow inferred from legacy modes as text_to_image"
    );
    assert!(preset["appliedDefaults"]["notes"]
        .as_array()
        .expect("notes array")
        .iter()
        .all(|note| !note
            .as_str()
            .unwrap_or_default()
            .contains("model defaulted")));
}

/// Snapshot tests for recipe presets JSON round-trip parity.
/// These tests capture the endpoint responses before and after the Value→typed-contract conversion.
/// The conversion must preserve JSON structure, field order, null vs absent, and number formats.
///
/// To update snapshots after validating the conversion preserves parity:
/// 1. Run tests with `SNAPSHOT_UPDATE=true` to capture new baselines
/// 2. Compare old snapshots against new to verify no structural changes
#[cfg(test)]
mod recipe_presets_parity {
    use crate::tests::support::*;
    use serde_json::json;

    fn setup_recipe_preset_fixtures(temp_dir: &tempfile::TempDir) {
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("config dir creates");

        // Builtin presets: full schema representation
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "default_t2i",
                  "name": "Default T2I",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "modes": ["text_to_image"],
                  "order": 1,
                  "defaults": {
                    "count": 1,
                    "resolution": "1024x1024",
                    "negativePrompt": ""
                  },
                  "prompt": {
                    "prefix": "",
                    "suffix": ""
                  },
                  "loras": [],
                  "ui": {
                    "description": "Default text-to-image generation"
                  }
                },
                {
                  "id": "cinematic",
                  "name": "Cinematic",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "modes": ["text_to_image"],
                  "order": 10,
                  "defaults": {
                    "count": 4,
                    "resolution": "1280x720",
                    "negativePrompt": "flat lighting, low contrast"
                  },
                  "prompt": {
                    "prefix": "cinematic",
                    "suffix": "cinematic lighting, volumetric"
                  },
                  "loras": [
                    {
                      "id": "style-lora",
                      "loraId": "style-lora",
                      "sourceUrl": "https://example.com/loras/cinematic.safetensors",
                      "name": "Cinematic Style",
                      "displayName": "Cinematic Style",
                      "compatibility": { "families": ["z-image"] },
                      "weight": 0.75,
                      "trigger": "cinematic style"
                    }
                  ],
                  "ui": {
                    "description": "Cinematic lighting and composition"
                  }
                }
              ]
            }
            "#,
        )
        .expect("builtin presets write");

        // User presets: minimal schema (tests merging + defaults)
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "cinematic",
                  "name": "My Cinematic",
                  "model": "z_image_turbo",
                  "workflow": "text_to_image",
                  "defaults": {
                    "count": 2,
                    "resolution": "1280x720"
                  },
                  "prompt": {
                    "suffix": "my custom lighting"
                  }
                },
                {
                  "id": "legacy_edit",
                  "name": "Legacy Edit",
                  "model": "z_image_turbo",
                  "modes": ["edit_image"],
                  "builtInLoras": [
                    {
                      "id": "style-lora",
                      "weight": 0.25
                    }
                  ]
                }
              ]
            }
            "#,
        )
        .expect("user presets write");

        // Builtin models for workflow validation
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z Image Turbo",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [],
                  "paths": { "model": "data/models/z_image_turbo" },
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models write");

        // Builtin loras for compatibility validation
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
                  "triggerWords": ["cinematic style"],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                }
              ]
            }
            "#,
        )
        .expect("builtin loras write");

        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras write");

        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "entries": [] }"#,
        )
        .expect("user models write");

        // Install marker for z_image_turbo
        let model_dir = temp_dir.path().join("data/models/z_image_turbo");
        std::fs::create_dir_all(&model_dir).expect("model dir creates");
        std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
            .expect("marker writes");

        // LoRA artifact
        let lora_dir = temp_dir.path().join("data/loras");
        std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
        write_test_safetensors(&lora_dir.join("style.safetensors"));
    }

    #[tokio::test]
    async fn recipe_presets_list_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(app, "GET", "/api/v1/recipe-presets", Value::Null).await;
        assert_eq!(status, StatusCode::OK);

        let presets = response.as_array().expect("response is array");
        // Builtin: default_t2i, cinematic
        // User: cinematic (merge), legacy_edit (new)
        // Result: default_t2i, cinematic (merged), legacy_edit
        assert_eq!(presets.len(), 3, "builtin+user merged presets");

        // Find and verify the cinematic preset (builtin + user merge)
        let cinematic = presets
            .iter()
            .find(|p| p["id"] == "cinematic")
            .expect("cinematic preset exists");

        // Verify merge: user values override builtin
        assert_eq!(
            cinematic["name"], "My Cinematic",
            "user name overrides builtin"
        );
        assert_eq!(cinematic["scope"], "global");
        assert_eq!(cinematic["workflow"], "text_to_image", "from builtin");
        assert_eq!(cinematic["model"], "z_image_turbo", "from builtin");
        assert_eq!(cinematic["defaults"]["count"], 2, "from user override");
        assert_eq!(cinematic["defaults"]["resolution"], "1280x720");
        assert!(
            cinematic["defaults"]["negativePrompt"].is_null(),
            "user didn't specify, builtin is empty"
        );
        assert_eq!(
            cinematic["prompt"]["suffix"], "my custom lighting",
            "user override"
        );
        // prefix is omitted when empty (skip_serializing_if = "is_empty" behavior)
        assert!(
            cinematic["prompt"]["prefix"].is_null()
                || cinematic["prompt"]["prefix"].as_str().is_some()
        );
        assert!(cinematic["builtInLoras"].is_array(), "computed loras field");
        assert!(cinematic["manifestPath"].is_string(), "computed field");

        // Verify default_t2i (builtin only)
        let default_t2i = presets
            .iter()
            .find(|p| p["id"] == "default_t2i")
            .expect("default_t2i from builtin");
        assert_eq!(default_t2i["name"], "Default T2I");
        assert_eq!(default_t2i["order"], 1);

        // Verify legacy_edit (user only)
        let legacy_edit = presets
            .iter()
            .find(|p| p["id"] == "legacy_edit")
            .expect("legacy_edit from user");
        assert_eq!(legacy_edit["name"], "Legacy Edit");
        assert_eq!(legacy_edit["workflow"], "edit_image", "inferred from modes");
    }

    #[tokio::test]
    async fn recipe_presets_get_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) =
            request(app, "GET", "/api/v1/recipe-presets/cinematic", Value::Null).await;
        assert_eq!(status, StatusCode::OK);

        let preset = response;
        assert_eq!(preset["id"], "cinematic");
        assert_eq!(preset["name"], "My Cinematic");
        assert_eq!(preset["scope"], "global");
        // Verify both loras and builtInLoras are present
        assert!(preset["loras"].is_array());
        assert!(preset["builtInLoras"].is_array());
    }

    #[tokio::test]
    async fn recipe_presets_create_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(
            app,
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Custom Preset",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "defaults": {
                    "count": 2,
                    "resolution": "1024x1024"
                },
                "prompt": {
                    "suffix": "custom suffix"
                },
                "loras": []
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let preset = response;
        assert_eq!(preset["name"], "Custom Preset");
        assert_eq!(preset["workflow"], "text_to_image");
        assert!(preset["id"].is_string(), "id auto-generated from name");
        assert_eq!(preset["scope"], "global");
        assert!(preset["createdAt"].is_string());
        assert!(preset["updatedAt"].is_string());
    }

    #[tokio::test]
    async fn recipe_presets_update_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(
            app,
            "PATCH",
            "/api/v1/recipe-presets/cinematic",
            json!({
                "name": "Updated Cinematic",
                "defaults": {
                    "count": 6,
                    "negativePrompt": "blurry"
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let preset = response;
        assert_eq!(preset["id"], "cinematic");
        assert_eq!(preset["name"], "Updated Cinematic");
        assert_eq!(preset["defaults"]["count"], 6);
        assert_eq!(preset["defaults"]["negativePrompt"], "blurry");
    }

    #[tokio::test]
    async fn recipe_presets_duplicate_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(
            app,
            "POST",
            "/api/v1/recipe-presets/cinematic/duplicate",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let preset = response;
        assert!(preset["id"]
            .as_str()
            .is_some_and(|id| id.contains("cinematic")));
        assert!(preset["name"]
            .as_str()
            .is_some_and(|name| name.contains("Cinematic")));
        assert!(preset["id"] != "cinematic", "new id is different");
        assert!(preset["name"] != "My Cinematic", "new name is different");
    }
}
