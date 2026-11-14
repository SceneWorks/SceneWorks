//! rust-api dataset batch describe tests (SW-7082, SW-7083): job enqueue, item subset in dataset
//! order, default/custom prompt carriage, JoyCaption as the second describer, and the 400s for an
//! unsupported model, an unknown item and an oversized prompt.
use super::support::*;
use sceneworks_core::dataset_batch_describe::{
    JOYCAPTION_REPO, MAX_DESCRIBE_PROMPT_CHARS, MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS,
    QWEN38_27B_DESCRIBER_MODEL_ID, QWEN38_27B_DESCRIBER_REPO, VISION_CAPTIONER_REPO,
};

fn png_rgb(w: u32, h: u32, shade: u8) -> Vec<u8> {
    let mut out = Vec::new();
    image::RgbImage::from_pixel(w, h, image::Rgb([shade, 64, 128]))
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode image");
    out
}

/// A project with one dataset of three images. Returns (app, project id, dataset id, item ids,
/// item relative paths, app state).
async fn dataset_fixture(
    temp_dir: &tempfile::TempDir,
) -> (
    axum::Router,
    String,
    String,
    Vec<String>,
    Vec<String>,
    crate::AppState,
) {
    let (app, state) = create_app_with_state(test_settings(temp_dir)).expect("app creates");
    let (_, project) = create_project(app.clone(), &"Batch describe project").await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let mut items = Vec::new();
    for index in 0..3u8 {
        let (status, upload) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/training/uploads"),
            &format!("image_{index}.png"),
            "image/png",
            &png_rgb(16, 8, index * 60),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{upload}");
        items.push(json!({
            "path": upload["file"]["path"],
            "displayName": format!("image_{index}.png"),
        }));
    }
    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Describe set", "items": items }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{dataset}");
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let field = |key: &str| -> Vec<String> {
        dataset["items"]
            .as_array()
            .expect("items")
            .iter()
            .map(|item| item[key].as_str().expect("item field").to_owned())
            .collect()
    };
    (
        app,
        project_id,
        dataset_id,
        field("id"),
        field("path"),
        state,
    )
}

#[tokio::test]
async fn batch_describe_job_enqueues_every_image_with_the_vision_captioner_by_default() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, paths, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, job) = request(app.clone(), "POST", &route, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "dataset_batch_describe");
    assert_eq!(job["title"], "Describe Dataset Images — Describe set");
    let payload = &job["payload"];
    assert_eq!(payload["kind"], "dataset_batch_describe");
    assert_eq!(payload["model"], VISION_CAPTIONER_REPO);
    assert_eq!(payload["datasetId"], dataset_id.as_str());
    assert_eq!(payload["datasetName"], "Describe set");
    assert!(
        payload.get("prompt").is_none(),
        "no prompt means the worker's default instruction"
    );
    let job_items = payload["items"].as_array().expect("job items");
    assert_eq!(job_items.len(), 3);
    for ((job_item, item_id), path) in job_items.iter().zip(&item_ids).zip(&paths) {
        assert_eq!(job_item["itemId"], item_id.as_str());
        assert_eq!(job_item["relativePath"], path.as_str());
        assert!(job_item["imagePath"]
            .as_str()
            .unwrap()
            .ends_with(path.as_str()));
    }
}

#[tokio::test]
async fn batch_describe_subset_keeps_dataset_order_and_carries_the_custom_prompt() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({
            "model": VISION_CAPTIONER_REPO,
            "prompt": "  Describe only the clothing.  ",
            // Asked for out of order: the job still describes them in dataset order.
            "itemIds": [item_ids[2], item_ids[0]],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["prompt"], "Describe only the clothing.");
    let ids: Vec<_> = job["payload"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["itemId"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(ids, vec![item_ids[0].clone(), item_ids[2].clone()]);

    // A blank prompt is the default instruction, not an empty one.
    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "prompt": "   ", "itemIds": [item_ids[1]] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(job["payload"].get("prompt").is_none());
    assert_eq!(job["payload"]["items"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn batch_describe_refuses_other_models_unknown_items_and_oversized_prompts() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, _, _, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");
    for (body, field, expect) in [
        (
            json!({ "model": "someone/other-vlm" }),
            "model",
            "Unsupported describe model",
        ),
        (
            json!({ "model": "fancyfeast/llama-joycaption-alpha-two-hf-llava" }),
            "model",
            "Unsupported describe model",
        ),
        (
            json!({ "model": QWEN38_27B_DESCRIBER_REPO, "variant": "q2" }),
            "variant",
            "Unsupported describe tier",
        ),
        (
            json!({ "variant": "q4" }),
            "variant",
            "not available for this model",
        ),
        (
            json!({ "itemIds": ["not_an_item"] }),
            "itemIds",
            "Unknown dataset item id(s): not_an_item.",
        ),
        (
            json!({ "itemIds": [] }),
            "itemIds",
            "at least one dataset item",
        ),
        (
            json!({ "prompt": "x".repeat(MAX_DESCRIBE_PROMPT_CHARS + 1) }),
            "prompt",
            "at most",
        ),
        (
            json!({
                "model": JOYCAPTION_REPO,
                "prompt": "x".repeat(MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS + 1),
            }),
            "prompt",
            "at most 3000 characters",
        ),
    ] {
        let (status, error) = request(app.clone(), "POST", &route, body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {error}");
        assert_eq!(error["context"]["field"], field, "{error}");
        assert!(
            error["detail"]
                .as_str()
                .unwrap_or_default()
                .contains(expect)
                || error["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains(expect)
                || error.to_string().contains(expect),
            "{error}"
        );
    }
}

#[tokio::test]
async fn batch_describe_accepts_joycaption_with_its_own_prompt_limit() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "model": format!("  {JOYCAPTION_REPO} "), "itemIds": [item_ids[1]] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "dataset_batch_describe");
    assert_eq!(job["payload"]["model"], JOYCAPTION_REPO);
    assert!(job["payload"].get("prompt").is_none());
    assert_eq!(job["payload"]["items"].as_array().unwrap().len(), 1);

    // An instruction exactly at JoyCaption's limit is accepted.
    let at_limit = "x".repeat(MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS);
    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "model": JOYCAPTION_REPO, "prompt": at_limit }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["prompt"].as_str().unwrap().chars().count(),
        MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS
    );
}

/// SW-7119: the Qwen3.8 27B describer is the third accepted model. Its job carries the tier (the
/// default when the request names none) and it takes the Vision Captioner's prompt limit.
#[tokio::test]
async fn batch_describe_accepts_the_qwen38_27b_describer_with_a_tier() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, _, _, state) = dataset_fixture(&temp_dir).await;
    let _hub = private_hub(&state);
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");
    for tier in ["q4", "q8", "bf16"] {
        seed_qwen38_tier(&state.settings.data_dir, tier);
    }
    for (tier, expect) in [(None, "q4"), (Some("q8"), "q8"), (Some("bf16"), "bf16")] {
        let mut body = json!({ "model": QWEN38_27B_DESCRIBER_REPO });
        if let Some(tier) = tier {
            body["variant"] = json!(tier);
        }
        let (status, job) = request(app.clone(), "POST", &route, body).await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        assert_eq!(job["payload"]["model"], QWEN38_27B_DESCRIBER_REPO);
        assert_eq!(job["payload"]["variant"], expect);
    }
    // The other describers have no tier, so their jobs carry none.
    let (status, job) = request(app.clone(), "POST", &route, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(job["payload"].get("variant").is_none());

    let (status, body) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/training/describe-prompt/default?model={QWEN38_27B_DESCRIBER_REPO}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["maxChars"], MAX_DESCRIBE_PROMPT_CHARS);
}

/// SW-7082: the describe worker appends each finished image to `/result-items` (its progress
/// result carries only counts), and the dialog pages them back in dataset order. Only the owning
/// worker may append, and a canceled job still serves the images it finished.
///
/// The describer is served on both native lanes (typed_registry_contract_tests pins the
/// `dataset_batch_describe` cells), so the claim runs once per worker host shape whatever OS
/// runs the test: the MLX/macOS worker and the Candle worker on Linux and Windows. A Candle
/// worker advertises its `candle` lane capability, as the shipped Candle worker does.
#[tokio::test]
async fn batch_describe_items_are_appended_by_the_owner_and_paged_back_in_order() {
    for (route, lane) in [
        (TestPortableRoute::MlxMacos, "mlx"),
        (TestPortableRoute::CandleLinux, "candle"),
        (TestPortableRoute::CandleWindows, "candle"),
    ] {
        batch_describe_items_round_trip_on(route, lane).await;
    }
}

async fn batch_describe_items_round_trip_on(worker_route: TestPortableRoute, lane: &str) {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _, state) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");
    let (status, job) = request(app.clone(), "POST", &route, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let job_id = job["id"].as_str().expect("job id").to_owned();
    const WORKER: &str = "test-describe-worker";
    // Admission froze every native route at creation; the claim runs on the one this worker is.
    assert!(
        state
            .jobs_store
            .portable_model_binding(&job_id)
            .expect("binding reads")
            .is_some(),
        "the describe job is bound when it is created"
    );
    claim_job_as_worker_on_route(
        &app,
        &job_id,
        WORKER,
        &["gpu", "prompt_refine", lane],
        worker_route,
    )
    .await;

    let items_route = format!("/api/v1/jobs/{job_id}/result-items");
    let item = |index: usize| {
        json!({
            "position": index,
            "item": { "itemId": item_ids[index], "description": format!("image {index}") },
        })
    };
    // Appended out of order; read back by position.
    for index in [1, 0] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &items_route,
            json!({ "workerId": WORKER, "items": [item(index)] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let (status, body) = request(
        app.clone(),
        "POST",
        &items_route,
        json!({ "workerId": "someone-else", "items": [item(2)] }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let (status, body) = request(
        app.clone(),
        "POST",
        &items_route,
        json!({ "workerId": WORKER, "items": [{ "position": 2, "item": { "description": "x" } }] }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an item needs an id: {body}"
    );

    let (status, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "workerId": WORKER, "status": "canceled", "stage": "canceled", "progress": 1.0,
            "message": "canceled", "result": { "processedCount": 2, "itemCount": 3 },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, page) = request(
        app.clone(),
        "GET",
        &format!("{items_route}?offset=0&limit=1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page}");
    assert_eq!(page["total"], 2);
    assert_eq!(page["nextOffset"], 1);
    assert_eq!(page["items"][0]["itemId"], item_ids[0].as_str());
    let (_, page) = request(
        app.clone(),
        "GET",
        &format!("{items_route}?offset=1"),
        Value::Null,
    )
    .await;
    assert_eq!(page["items"][0]["itemId"], item_ids[1].as_str());
    assert_eq!(page["items"][0]["description"], "image 1");
    assert_eq!(page["nextOffset"], Value::Null);

    let (status, _) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs/missing-job/result-items",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn batch_describe_jobs_list_the_datasets_runs_newest_first() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, item_ids, _, _) = dataset_fixture(&temp_dir).await;
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");

    let (status, empty) = request(app.clone(), "GET", &route, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{empty}");
    assert_eq!(empty, json!([]));

    let (status, first) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "itemIds": [item_ids[0]] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{first}");
    // Each payload item records the image's identity when it was described.
    assert!(first["payload"]["items"][0]
        .as_object()
        .expect("payload item")
        .contains_key("contentHash"));
    let (status, second) = request(app.clone(), "POST", &route, json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{second}");
    // A canceled run still lists: its stored descriptions are the user's work.
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/cancel", first["id"].as_str().unwrap()),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{canceled}");

    let (status, jobs) = request(app.clone(), "GET", &route, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{jobs}");
    let ids: Vec<_> = jobs
        .as_array()
        .expect("job list")
        .iter()
        .map(|job| job["id"].clone())
        .collect();
    assert_eq!(ids, vec![second["id"].clone(), first["id"].clone()]);

    // Only the unfinished job: the canceled one is left out.
    let (status, unfinished) = request(
        app.clone(),
        "GET",
        &format!("{route}?unfinished=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{unfinished}");
    assert_eq!(unfinished.as_array().map(|jobs| jobs.len()), Some(1));
    assert_eq!(unfinished[0]["id"], second["id"]);

    // Paged: offset skips the newest jobs, limit caps the page.
    let (status, page) = request(
        app.clone(),
        "GET",
        &format!("{route}?offset=1&limit=1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page}");
    assert_eq!(page.as_array().map(|jobs| jobs.len()), Some(1));
    assert_eq!(page[0]["id"], first["id"]);

    // Another dataset's history is separate.
    let other = format!(
        "/api/v1/projects/{project_id}/training/datasets/other-dataset/batch-describe-jobs"
    );
    let (status, none) = request(app.clone(), "GET", &other, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{none}");
    assert_eq!(none, json!([]));
}

// ── Qwen3.8 27B tiers: install, source binding, routing and refusal (SW-7121) ───────────────

/// The catalog row of the 27B's `tier` on `platform`.
fn qwen38_tier_row(platform: &str, tier: &str) -> Value {
    let mut entry = Value::Object(
        sceneworks_core::builtin_manifests::builtin_model_entry(QWEN38_27B_DESCRIBER_MODEL_ID)
            .expect("the 27B describer is a builtin")
            .clone(),
    );
    crate::models::retain_downloads_for_os(&mut entry, platform);
    entry["downloads"]
        .as_array()
        .expect("downloads")
        .iter()
        .find(|row| row["variant"] == tier)
        .cloned()
        .unwrap_or_else(|| panic!("{platform} has a {tier} row"))
}

/// Give this app its own Hugging Face cache, so tiers staged by a test running alongside never
/// read as installed here. Hold the guard for the whole test.
fn private_hub(state: &crate::AppState) -> sceneworks_core::hf_home::test_support::CacheRootGuard {
    sceneworks_core::hf_home::test_support::CacheRootGuard::new(
        state.settings.data_dir.clone(),
        state.settings.data_dir.join("cache/huggingface/hub"),
    )
}

/// Stage `tier` of the 27B in this host's Hugging Face cache, one file per selector of its row,
/// exactly where a finished download of that tier leaves it.
fn seed_qwen38_tier(data_dir: &std::path::Path, tier: &str) {
    let row = qwen38_tier_row(std::env::consts::OS, tier);
    let repo = row["repo"].as_str().expect("repo");
    let revision = row["revision"].as_str().expect("revision");
    let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
    std::fs::create_dir_all(cache.join("refs")).expect("refs dir");
    std::fs::write(cache.join("refs/main"), revision).expect("ref");
    for selector in row["files"].as_array().expect("files") {
        let file = selector.as_str().expect("selector").replace('*', "fixture");
        let path = cache.join("snapshots").join(revision).join(&file);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("snapshot dir");
        // A shard index names the one staged shard, so the sharded tier reads complete.
        let body = if file.ends_with(".safetensors.index.json") {
            json!({ "weight_map": { "w": "model-fixture.safetensors" } }).to_string()
        } else {
            "fixture".to_owned()
        };
        std::fs::write(path, body).expect("fixture file");
    }
}

/// The pinned source the job's binding froze for the describer on `backend:platform`:
/// (repository, revision, variant, every frozen file).
fn bound_describer_source(
    state: &crate::AppState,
    job_id: &str,
    backend: &str,
    platform: &str,
) -> (String, String, String, Vec<String>) {
    use sceneworks_core::model_artifacts::portable_binding::{
        PortableJobBinding, PortableSourceIdentity,
    };
    let binding: PortableJobBinding = serde_json::from_value(
        state
            .jobs_store
            .portable_model_binding(job_id)
            .expect("binding reads")
            .expect("the job is bound at creation"),
    )
    .expect("binding parses");
    let route = binding
        .select_route(backend, platform)
        .unwrap_or_else(|error| panic!("{backend}:{platform} is admitted: {error}"));
    assert_eq!(route.models.len(), 1, "{backend}:{platform}");
    let model = &route.models[0];
    assert_eq!(model.model_id, QWEN38_27B_DESCRIBER_MODEL_ID);
    assert_eq!(model.members.len(), 1, "one source: the tier's own row");
    let member = &model.members[0];
    let PortableSourceIdentity::Revision { identity } = &member.source else {
        panic!("a catalog tier is a pinned revision: {:?}", member.source);
    };
    (
        identity.repository.clone(),
        identity.revision.clone(),
        identity.variant.clone(),
        member
            .files
            .iter()
            .map(|file| file.relative_path.to_string_lossy().into_owned())
            .collect(),
    )
}

/// The recorded defect: a q4/q8 job bound the full-precision upstream row on every tier, so the
/// re-host folder the worker actually loads was verified by nothing. Each route of a job now
/// freezes the row of the job's tier on that route's platform: the MLX re-host on macOS, the Candle
/// re-host on Linux and Windows, and the upstream repository for full precision.
#[tokio::test]
async fn a_qwen38_job_binds_its_tiers_row_on_every_route() {
    use sceneworks_core::dataset_batch_describe::{
        QWEN38_27B_DESCRIBER_CANDLE_REPO, QWEN38_27B_DESCRIBER_MLX_REPO,
    };
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, _, _, state) = dataset_fixture(&temp_dir).await;
    let _hub = private_hub(&state);
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");
    for tier in ["q4", "q8", "bf16"] {
        seed_qwen38_tier(&state.settings.data_dir, tier);
        let (status, job) = request(
            app.clone(),
            "POST",
            &route,
            json!({ "model": QWEN38_27B_DESCRIBER_REPO, "variant": tier }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{tier}: {job}");
        let job_id = job["id"].as_str().expect("job id");
        for (backend, platform, repo) in [
            ("mlx", "macos", QWEN38_27B_DESCRIBER_MLX_REPO),
            ("candle", "linux", QWEN38_27B_DESCRIBER_CANDLE_REPO),
            ("candle", "windows", QWEN38_27B_DESCRIBER_CANDLE_REPO),
        ] {
            let (bound_repo, revision, variant, files) =
                bound_describer_source(&state, job_id, backend, platform);
            let row = qwen38_tier_row(platform, tier);
            assert_eq!(variant, tier, "{backend}:{platform}");
            assert_eq!(revision, row["revision"], "{backend}:{platform} {tier}");
            if tier == "bf16" {
                assert_eq!(bound_repo, QWEN38_27B_DESCRIBER_REPO, "{platform}");
            } else {
                assert_eq!(bound_repo, repo, "{platform} {tier}");
                assert!(
                    !files.is_empty()
                        && files
                            .iter()
                            .all(|file| file.starts_with(&format!("{tier}/"))),
                    "{platform} {tier} freezes only its own folder: {files:?}"
                );
            }
        }
    }
}

/// AC2: the job routes to a capable worker on each lane, through the binding admission froze
/// (no fixture binding): the MLX worker on macOS and the Candle worker on Linux and Windows.
#[tokio::test]
async fn a_qwen38_job_is_claimed_by_a_capable_worker_on_each_lane() {
    for (worker_route, lane) in [
        (TestPortableRoute::MlxMacos, "mlx"),
        (TestPortableRoute::CandleLinux, "candle"),
        (TestPortableRoute::CandleWindows, "candle"),
    ] {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let (app, project_id, dataset_id, _, _, state) = dataset_fixture(&temp_dir).await;
        let _hub = private_hub(&state);
        seed_qwen38_tier(&state.settings.data_dir, "q4");
        let (status, job) = request(
            app.clone(),
            "POST",
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs"
            ),
            json!({ "model": QWEN38_27B_DESCRIBER_REPO }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        claim_job_as_worker_on_route(
            &app,
            job["id"].as_str().expect("job id"),
            "test-27b-describe-worker",
            &["gpu", "prompt_refine", lane],
            worker_route,
        )
        .await;
    }
}

/// AC3 (install half): a tier this host has not installed is refused before a job exists, with a
/// message that names the tier and what to do. Another installed tier does not stand in for it.
#[tokio::test]
async fn a_qwen38_tier_that_is_not_installed_is_refused_before_queueing() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id, dataset_id, _, _, state) = dataset_fixture(&temp_dir).await;
    let _hub = private_hub(&state);
    let route =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs");
    seed_qwen38_tier(&state.settings.data_dir, "q4");
    for tier in ["q8", "bf16"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &route,
            json!({ "model": QWEN38_27B_DESCRIBER_REPO, "variant": tier }),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{tier}: {body}");
        let detail = body["detail"].as_str().expect("detail");
        assert!(
            detail.starts_with(&format!(
                "The {tier} tier of Vision Describer (Qwen3.8-27B) is not installed."
            )),
            "{detail}"
        );
        assert!(detail.contains("Model Manager"), "{detail}");
    }
    let (status, jobs) = request(app.clone(), "GET", &route, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{jobs}");
    assert_eq!(jobs, json!([]), "a refused request queues nothing");
    // The installed tier (the default) is accepted.
    let (status, job) = request(
        app.clone(),
        "POST",
        &route,
        json!({ "model": QWEN38_27B_DESCRIBER_REPO }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
}

/// AC1: installing the q4 tier downloads the q4 folder of this platform's re-host and nothing
/// else (never the full-precision files), and it is the same pinned source a q4 describe job binds
/// for this platform's worker, so the installed tier is the one the worker loads.
#[tokio::test]
async fn installing_the_q4_tier_downloads_only_the_q4_files_the_q4_job_loads() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    // The catalog this app serves holds the shipped 27B row (its tiers come from the registry).
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let entry =
        sceneworks_core::builtin_manifests::builtin_model_entry(QWEN38_27B_DESCRIBER_MODEL_ID)
            .expect("the 27B describer is a builtin");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [entry] }).to_string(),
    )
    .expect("builtin models writes");
    write_empty_sibling_manifests(&config_dir);
    let (app, project_id, dataset_id, _, _, state) = dataset_fixture(&temp_dir).await;
    let _hub = private_hub(&state);
    let (status, download) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/models/{QWEN38_27B_DESCRIBER_MODEL_ID}/download"),
        json!({ "variant": "q4" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{download}");
    let payload = &download["payload"];
    let row = qwen38_tier_row(std::env::consts::OS, "q4");
    assert_eq!(payload["repo"], row["repo"]);
    assert_ne!(payload["repo"], QWEN38_27B_DESCRIBER_REPO);
    assert_eq!(payload["revision"], row["revision"]);
    assert_eq!(payload["files"], json!(["q4/*"]));
    assert_eq!(payload["variant"], "q4");

    seed_qwen38_tier(&state.settings.data_dir, "q4");
    let (status, job) = request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-describe-jobs"
        ),
        json!({ "model": QWEN38_27B_DESCRIBER_REPO, "variant": "q4" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let (backend, platform) = if cfg!(target_os = "macos") {
        ("mlx", "macos")
    } else {
        ("candle", std::env::consts::OS)
    };
    let (repo, revision, variant, files) = bound_describer_source(
        &state,
        job["id"].as_str().expect("job id"),
        backend,
        platform,
    );
    assert_eq!(repo, payload["repo"].as_str().expect("repo"));
    assert_eq!(revision, payload["revision"].as_str().expect("revision"));
    assert_eq!(variant, "q4");
    assert!(
        files.iter().all(|file| file.starts_with("q4/")),
        "{files:?}"
    );
}
