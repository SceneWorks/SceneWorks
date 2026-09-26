//! sc-22998: the YuE2 catalog entry through the real HTTP routes — its install queues exactly the
//! upstream generation closure (model + tokenizer + the chosen decoder), its licence gate holds, a
//! commercial-use pointer resolves to YuE1, and nothing crosses between the YuE1 and YuE2 families.
//!
//! The YuE2 entry is the LIVE builtin row (read from the embedded manifest, not re-typed here). The
//! YuE1 entries are fixtures in the shape the YuE1 epic (sc-19373) ships, which is not on this branch
//! (epic sc-22988 acceptance test 5).
use super::support::*;

fn builtin_yue2() -> Value {
    let (_, contents) = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc is embedded");
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(contents))
            .expect("builtin manifest parses");
    manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["id"] == "yue2")
        .expect("yue2 is in the builtin catalog")
        .clone()
}

fn yue1_fixture(id: &str) -> Value {
    json!({
        "id": id,
        "name": format!("YuE {id}"),
        "family": "yue",
        "type": "audio",
        "downloads": [{
            "provider": "huggingface",
            "repo": format!("SceneWorks/{}-candle", id.replace('_', "-")),
            "revision": "5842b8bc97d2a6dddc427a444920050dd3860949",
            "variant": "q4",
            "default": true,
            "files": ["q4/*"]
        }]
    })
}

fn app_with_yue1_and_yue2(temp_dir: &tempfile::TempDir) -> axum::Router {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let manifest = json!({
        "schemaVersion": 1,
        "models": [yue1_fixture("yue_en_cot"), builtin_yue2(), yue1_fixture("yue_zh_icl")]
    });
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .expect("builtin models writes");
    write_empty_sibling_manifests(&config_dir);
    create_app(test_settings(temp_dir)).expect("app creates")
}

async fn queued_downloads(app: axum::Router) -> Vec<Value> {
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    jobs.as_array()
        .expect("jobs is an array")
        .iter()
        .filter(|job| job["type"] == "model_download")
        .map(|job| job["payload"].clone())
        .collect()
}

fn repos(payloads: &[Value]) -> std::collections::BTreeSet<String> {
    payloads
        .iter()
        .map(|payload| payload["repo"].as_str().unwrap().to_owned())
        .collect()
}

#[tokio::test]
async fn yue2_install_is_licence_gated_and_queues_the_model_and_one_decoder_only() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = app_with_yue1_and_yue2(&temp_dir);

    // CC BY-NC 4.0 terms must be accepted before any byte is fetched.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/models/yue2/download",
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(queued_downloads(app.clone()).await.is_empty());

    let (status, primary) = request(
        app.clone(),
        "POST",
        "/api/v1/models/yue2/download",
        json!({ "requestedGpu": "auto", "licenseAcknowledged": true }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{primary}");
    assert_eq!(primary["payload"]["repo"], "m-a-p/YuE2-3B");
    assert_eq!(primary["payload"]["variant"], "bf16");
    assert_eq!(
        primary["payload"]["revision"],
        "1a96eca688d6ae5d7f0feb88573fec89920fcd19"
    );
    assert!(primary["payload"]["files"]
        .as_array()
        .unwrap()
        .contains(&json!("qwen.tiktoken")));
    let queued = queued_downloads(app).await;
    // Mutation that reds this: dropping the choice filter queues YuE2-Vae-legacy too; moving a
    // cover dependency into `downloads` queues SheetSage2 / MERT.
    assert_eq!(
        repos(&queued),
        ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]
            .map(str::to_owned)
            .into_iter()
            .collect()
    );
}

#[tokio::test]
async fn yue2_install_honours_a_chosen_decoder_and_refuses_an_unknown_one() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = app_with_yue1_and_yue2(&temp_dir);

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/models/yue2/download",
        json!({ "licenseAcknowledged": true, "choices": { "decoder": "fp16" } }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(queued_downloads(app.clone()).await.is_empty());

    let (status, primary) = request(
        app.clone(),
        "POST",
        "/api/v1/models/yue2/download",
        json!({
            "licenseAcknowledged": true,
            "variant": "q4",
            "choices": { "decoder": "legacy" }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{primary}");
    // A q4 install fetches the bf16 ORIGINAL (the tier is derived locally), never a re-host.
    let bf16 = builtin_yue2()["downloads"][0].clone();
    assert_eq!(bf16["variant"], "bf16");
    assert_eq!(primary["payload"]["variant"], "q4");
    assert_eq!(primary["payload"]["repo"], bf16["repo"]);
    assert_eq!(primary["payload"]["revision"], bf16["revision"]);
    assert_eq!(primary["payload"]["files"], bf16["files"]);
    assert_eq!(
        repos(&queued_downloads(app).await),
        ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae-legacy"]
            .map(str::to_owned)
            .into_iter()
            .collect()
    );
}

#[tokio::test]
async fn yue1_and_yue2_installs_never_cross_namespaces() {
    // Swapped-model negative fixtures at the install route: a YuE1 id installs only its own
    // re-host, and a choice meant for YuE2 is refused on a YuE1 entry instead of being ignored.
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = app_with_yue1_and_yue2(&temp_dir);
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/models/yue_en_cot/download",
        json!({ "choices": { "decoder": "standard" } }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/models/yue_en_cot/download",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let queued = queued_downloads(app.clone()).await;
    assert_eq!(
        repos(&queued),
        ["SceneWorks/yue-en-cot-candle".to_owned()]
            .into_iter()
            .collect()
    );
    // An id neither family declares is a 404, not the nearest model.
    let (status, _) = request(
        app,
        "POST",
        "/api/v1/models/yue/download",
        json!({ "licenseAcknowledged": true }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_shared_download_body_refuses_choices_on_a_lora_instead_of_ignoring_them() {
    // `ModelDownloadRequest` is shared with LoRA downloads, which have no choice groups. Mutation
    // that reds this: drop the refusal in `create_lora_download_job` (the lookup then 404s).
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = app_with_yue1_and_yue2(&temp_dir);
    let (status, body) = request(
        app,
        "POST",
        "/api/v1/loras/any_lora/download",
        json!({ "choices": { "decoder": "legacy" } }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn catalog_points_commercial_use_from_yue2_to_the_yue1_entries() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = app_with_yue1_and_yue2(&temp_dir);
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let by_id = |id: &str| {
        models
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == id)
            .unwrap_or_else(|| panic!("{id} listed"))
            .clone()
    };
    let yue2 = by_id("yue2");
    assert_eq!(yue2["experimental"], true);
    assert_eq!(yue2["nonCommercial"], true);
    assert_eq!(yue2["requiresLicenseAcknowledgment"], true);
    assert_eq!(yue2["commercialUse"]["eligible"], false);
    assert_eq!(
        yue2["commercialUse"]["alternatives"],
        json!(["yue_en_cot", "yue_zh_icl"])
    );
    // YuE1 declares no restriction, so it carries no pointer and no V2 reference.
    let yue1 = by_id("yue_en_cot");
    assert!(yue1.get("commercialUse").is_none());
    assert!(!yue1.to_string().contains("YuE2"));
}

/// Seed a pinned snapshot under the test's own hub cache holding `files` (placeholder bytes).
fn seed_snapshot(temp_dir: &tempfile::TempDir, download: &Value) {
    let repo = download["repo"].as_str().unwrap().replace('/', "--");
    let snapshot = temp_dir
        .path()
        .join(format!(
            "data/cache/huggingface/hub/models--{repo}/snapshots"
        ))
        .join(download["revision"].as_str().unwrap());
    for file in download["files"].as_array().unwrap() {
        let path = snapshot.join(file.as_str().unwrap());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }
}

#[tokio::test]
async fn either_decoder_completes_a_yue2_install_and_neither_leaves_it_repairable() {
    // Mutation that reds this: gate install on every decoder option (the legacy-only install reads
    // incomplete) or on none (the decoder-less install reads installed).
    let yue2 = builtin_yue2();
    let row = |repo: &str| {
        yue2["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["repo"] == repo && row.get("coRequisite").is_none_or(|v| v != true))
            .or_else(|| {
                yue2["downloads"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|row| row["repo"] == repo)
            })
            .unwrap()
            .clone()
    };
    for (decoder, expect_installed) in [
        (None, false),
        (Some("m-a-p/YuE2-Vae-legacy"), true),
        (Some("m-a-p/YuE2-Vae"), true),
    ] {
        let _env = isolate_hf_cache();
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = app_with_yue1_and_yue2(&temp_dir);
        seed_snapshot(&temp_dir, &row("m-a-p/YuE2-3B"));
        if let Some(repo) = decoder {
            seed_snapshot(&temp_dir, &row(repo));
        }
        let (_, models) = request(app, "GET", "/api/v1/models", Value::Null).await;
        let entry = models
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == "yue2")
            .unwrap()
            .clone();
        assert_eq!(
            entry["installState"],
            json!(if expect_installed {
                "installed"
            } else {
                "missing"
            }),
            "decoder {decoder:?}: {entry}"
        );
        if !expect_installed {
            assert_eq!(entry["repairAvailable"], true, "{entry}");
            let missing = entry["missingRequiredFiles"].to_string();
            assert!(missing.contains("m-a-p/YuE2-Vae"), "{missing}");
            assert!(!missing.contains("YuE2-Vae-legacy"), "{missing}");
        }
    }
}
