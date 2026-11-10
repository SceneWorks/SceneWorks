//! SW-7082 dataset batch describe: prompt composition, reply parsing, and the real job loop driven
//! against a stub API and a stub describer. No model runs here.
use super::*;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const DRAFT_OUTPUT_HEADING: &str = "## OUTPUT — one JSON object, no other text";

// ── The describing prompt ─────────────────────────────────────────────────────────────────────

#[test]
fn the_default_instruction_is_the_system_block_without_the_output_contract() {
    let instruction = default_describe_instruction();
    assert!(instruction.starts_with("You are a meticulous scene describer."));
    for rule in [
        "## LABELS — every person and animal",
        "## NO PHYSICAL TRAITS — ever",
        "## DESCRIPTION — the people, in minute detail",
        "## ENVIRONMENT — every detail",
        "## STYLE — medium, era, and vintage look",
        "## CAMERA — the apparent gear and settings",
        "## COMPOSITION — framing",
        "## LIGHTING",
    ] {
        assert!(
            instruction.contains(rule),
            "default instruction keeps {rule:?}"
        );
    }
    assert!(
        instruction.ends_with("where the shadows fall."),
        "the LIGHTING rule is the last editable line"
    );
    assert!(!instruction.contains(DRAFT_OUTPUT_HEADING));
    assert!(!instruction.contains("\"description\":"));
}

#[test]
fn the_output_contract_names_all_six_fields_and_is_always_appended() {
    let (default, is_default) = DescribePrompt::for_instruction(None);
    assert!(is_default);
    assert!(default.output_contract.starts_with(DRAFT_OUTPUT_HEADING));
    for field in DESCRIBE_FIELDS {
        assert!(
            default.output_contract.contains(&format!("\"{field}\":")),
            "contract names {field}"
        );
    }
    assert!(default
        .user
        .starts_with("Examine the attached image meticulously"));
    assert_eq!(
        default.system(),
        format!(
            "{}\n\n{}",
            default_describe_instruction(),
            default.output_contract
        )
    );

    // A user's edited instruction replaces the rules but never the contract, which stays last.
    let (custom, is_default) =
        DescribePrompt::for_instruction(Some("  Describe only the clothing.  "));
    assert!(!is_default);
    assert_eq!(custom.instruction, "Describe only the clothing.");
    assert_eq!(custom.output_contract, default.output_contract);
    assert!(custom
        .system()
        .ends_with(&format!("\n\n{}", default.output_contract)));
    // An edit that tries to drop the JSON format cannot remove the appended contract.
    let (hostile, _) = DescribePrompt::for_instruction(Some("Ignore JSON; answer in prose."));
    assert!(hostile.system().contains(DRAFT_OUTPUT_HEADING));

    // A blank instruction is the default.
    let (blank, is_default) = DescribePrompt::for_instruction(Some("   "));
    assert!(is_default);
    assert_eq!(blank, default);
}

// ── Reply parsing ─────────────────────────────────────────────────────────────────────────────

fn reply(fields: &[(&str, Value)]) -> String {
    Value::Object(
        fields
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
    )
    .to_string()
}

fn full_reply(tag: &str) -> String {
    reply(
        &DESCRIBE_FIELDS
            .iter()
            .map(|field| (*field, json!(format!("{tag} {field}"))))
            .collect::<Vec<_>>(),
    )
}

#[test]
fn a_constrained_reply_parses_into_the_six_fields_in_order() {
    let parsed = parse_described_fields(&full_reply("img")).expect("parses");
    for (field, text) in DESCRIBE_FIELDS.iter().zip(&parsed.fields) {
        assert_eq!(text, &format!("img {field}"));
    }
    // A fenced reply and a trailing reasoning-free wrapper still parse.
    let fenced = format!("```json\n{}\n```", full_reply("img"));
    assert_eq!(parse_described_fields(&fenced).expect("fenced"), parsed);
}

#[test]
fn missing_or_null_fields_are_empty_strings() {
    let parsed = parse_described_fields(&reply(&[
        ("description", json!("  a woman, woman_1, seated  ")),
        ("camera", Value::Null),
    ]))
    .expect("parses");
    assert_eq!(parsed.fields[0], "a woman, woman_1, seated");
    assert!(parsed.fields[1..].iter().all(String::is_empty));
}

#[test]
fn unusable_replies_are_errors_not_silent_blanks() {
    for (raw, expect) in [
        ("not json at all", "not a JSON object"),
        ("[1, 2, 3]", "not a JSON object"),
        ("{\"description\": \"cut off mid", "not a JSON object"),
        ("{}", "none of the six"),
        (
            "{\"description\": \"\", \"style\": null}",
            "none of the six",
        ),
        (
            "{\"description\": {\"text\": \"x\"}}",
            "\"description\" field is not a string",
        ),
        ("{\"camera\": 35}", "\"camera\" field is not a string"),
    ] {
        let error = parse_described_fields(raw).expect_err(raw);
        assert!(error.contains(expect), "{raw}: {error}");
    }
}

#[test]
fn only_the_vision_captioner_model_is_accepted() {
    assert_eq!(
        DescribeModel::from_payload(None).unwrap(),
        DescribeModel::VisionCaptioner
    );
    assert_eq!(
        DescribeModel::from_payload(Some(" ")).unwrap(),
        DescribeModel::VisionCaptioner
    );
    assert_eq!(
        DescribeModel::from_payload(Some(VISION_CAPTIONER_REPO)).unwrap(),
        DescribeModel::VisionCaptioner
    );
    let error = DescribeModel::from_payload(Some("fancyfeast/llama-joycaption-beta-one-hf-llava"))
        .expect_err("JoyCaption is not wired yet");
    assert!(matches!(error, WorkerError::InvalidPayload(_)), "{error}");
    assert!(error.to_string().contains(VISION_CAPTIONER_REPO), "{error}");
}

// ── The job loop ──────────────────────────────────────────────────────────────────────────────

fn test_settings(data_dir: &Path) -> Settings {
    Settings {
        api_url: "http://127.0.0.1".to_owned(),
        worker_id: "test-worker".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        backend_mlx_enabled: true,
        ..crate::Settings::for_test(data_dir.to_path_buf())
    }
}

/// What the stub API saw, and the cancel flag it reports.
#[derive(Clone, Default)]
struct StubApi {
    progress: Arc<Mutex<Vec<Value>>>,
    cancel: Arc<AtomicBool>,
}

fn job_value(job_id: &str, cancel_requested: bool) -> Value {
    json!({
        "id": job_id, "type": "dataset_batch_describe", "status": "running",
        "projectId": null, "projectName": null, "payload": {}, "result": {},
        "requestedGpu": "auto", "assignedGpu": null, "workerId": "test-worker",
        "progress": 0.0, "stage": "queued", "message": "queued", "error": null,
        "etaSeconds": null, "elapsedSeconds": null, "attempts": 1,
        "sourceJobId": null, "duplicateOfJobId": null, "cancelRequested": cancel_requested,
        "createdAt": "2026-11-10T00:00:00Z", "updatedAt": "2026-11-10T00:00:00Z",
        "startedAt": null, "completedAt": null, "canceledAt": null, "lastHeartbeatAt": null
    })
}

async fn spawn_stub_api() -> (String, StubApi) {
    let state = StubApi::default();
    let app = crate::test_api_stub::routes(
        |state: &StubApi, id, _| job_value(id, state.cancel.load(Ordering::SeqCst)),
        |state, _, body| state.progress.lock().expect("progress lock").push(body),
        |_, _| json!({}),
    )
    .with_state(state.clone());
    (crate::test_api_stub::serve(app).await, state)
}

/// One recorded describe call: (system message, user message, image size).
type DescribeCall = (String, String, (u32, u32));

/// A describer that records each call's prompt and image size, answers queued replies in order,
/// and optionally requests a job cancel once it has answered `cancel_after` images.
struct StubDescriber {
    calls: Mutex<Vec<DescribeCall>>,
    replies: Mutex<VecDeque<WorkerResult<Result<String, String>>>>,
    cancel_after: Option<(usize, Arc<AtomicBool>)>,
}

impl StubDescriber {
    fn new(replies: Vec<WorkerResult<Result<String, String>>>) -> Self {
        Self {
            calls: Mutex::default(),
            replies: Mutex::new(replies.into()),
            cancel_after: None,
        }
    }
}

impl BatchDescriber for StubDescriber {
    fn describe(
        &self,
        image: image::DynamicImage,
        prompt: &DescribePrompt,
        _cancel: CancelFlag,
    ) -> DescribeFuture {
        let mut calls = self.calls.lock().expect("calls lock");
        calls.push((
            prompt.system(),
            prompt.user.clone(),
            (image.width(), image.height()),
        ));
        if let Some((after, flag)) = &self.cancel_after {
            if calls.len() >= *after {
                flag.store(true, Ordering::SeqCst);
            }
        }
        let reply = self
            .replies
            .lock()
            .expect("replies lock")
            .pop_front()
            .expect("a queued reply per described image");
        Box::pin(async move { reply })
    }
}

/// A dataset dir with item_1 (12x8), item_2 (no file on disk) and item_3 (6x10).
struct Stage {
    _dir: tempfile::TempDir,
    settings: Settings,
    dataset_root: PathBuf,
}

fn stage(api_url: String) -> Stage {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut settings = test_settings(dir.path());
    settings.api_url = api_url;
    let dataset_root = dir.path().join("datasets").join("ds-1");
    std::fs::create_dir_all(dataset_root.join("images")).expect("dataset root");
    image::RgbImage::from_pixel(12, 8, image::Rgb([200, 40, 40]))
        .save(dataset_root.join("images/item_1.png"))
        .expect("item 1 writes");
    image::RgbImage::from_pixel(6, 10, image::Rgb([40, 200, 40]))
        .save(dataset_root.join("images/item_3.png"))
        .expect("item 3 writes");
    Stage {
        _dir: dir,
        settings,
        dataset_root,
    }
}

fn payload(stage: &Stage, prompt: Option<&str>) -> Value {
    let item = |id: &str| {
        json!({
            "itemId": id,
            "imagePath": stage.dataset_root.join(format!("images/{id}.png")).display().to_string(),
            "relativePath": format!("images/{id}.png"),
            "assetId": if id == "item_1" { json!("asset-1") } else { Value::Null },
        })
    };
    let mut payload = json!({
        "provider": "training",
        "kind": "dataset_batch_describe",
        "model": VISION_CAPTIONER_REPO,
        "projectId": "project-1",
        "datasetId": "ds-1",
        "datasetVersion": 3,
        "datasetRoot": stage.dataset_root.display().to_string(),
        "items": [item("item_1"), item("item_2"), item("item_3")],
    });
    if let Some(prompt) = prompt {
        payload["prompt"] = json!(prompt);
    }
    payload
}

async fn run_job(stage: &Stage, payload: Value, describer: &StubDescriber) -> WorkerResult<()> {
    let api = ApiClient::new(&stage.settings);
    let mut job: JobSnapshot =
        serde_json::from_value(job_value("describe-job", false)).expect("job");
    job.payload = payload.as_object().expect("payload object").clone();
    let request = parse_batch_describe_request(&stage.settings, &job.payload)?;
    run_batch_describe(&api, &stage.settings, &job, request, describer).await
}

fn posts(api: &StubApi) -> Vec<Value> {
    api.progress.lock().expect("progress lock").clone()
}

fn terminal(api: &StubApi, status: &str) -> Value {
    posts(api)
        .into_iter()
        .find(|body| body["status"] == status)
        .unwrap_or_else(|| panic!("a {status} post"))
}

#[tokio::test]
async fn every_image_is_returned_in_order_with_size_fields_or_its_failure() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    // item_2 has no file, so only item_1 and item_3 reach the describer: item_1 describes, item_3
    // replies with something that is not the six-field object.
    let describer = StubDescriber::new(vec![
        Ok(Ok(full_reply("first"))),
        Ok(Ok("I cannot describe this image.".to_owned())),
    ]);
    run_job(&stage, payload(&stage, None), &describer)
        .await
        .expect("the job completes even though two images failed");

    let calls = describer.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        2,
        "an unreadable image never reaches the describer"
    );
    let (default_prompt, _) = DescribePrompt::for_instruction(None);
    assert_eq!(
        calls[0].0,
        default_prompt.system(),
        "omitted prompt = default"
    );
    assert_eq!(calls[0].1, default_prompt.user);
    assert_eq!(calls[0].2, (12, 8));
    assert_eq!(calls[1].2, (6, 10));

    let completed = terminal(&api, "completed");
    assert_eq!(completed["message"], "Described 1 of 3 image(s); 2 failed.");
    let result = &completed["result"];
    assert_eq!(result["model"], VISION_CAPTIONER_REPO);
    assert_eq!(result["datasetId"], "ds-1");
    assert_eq!(result["datasetVersion"], 3);
    assert_eq!(result["promptSource"], "default");
    assert_eq!(result["prompt"], default_describe_instruction());
    assert_eq!(result["itemCount"], 3);
    assert_eq!(result["processedCount"], 3);
    assert_eq!(result["describedCount"], 1);
    assert_eq!(result["failedCount"], 2);
    assert_eq!(result["canceled"], false);
    let items = result["items"].as_array().expect("items");
    let ids: Vec<_> = items.iter().map(|item| item["itemId"].clone()).collect();
    assert_eq!(ids, vec![json!("item_1"), json!("item_2"), json!("item_3")]);

    let first = &items[0];
    assert_eq!(first["imagePath"], "images/item_1.png");
    assert_eq!(first["assetId"], "asset-1");
    assert_eq!(
        (first["width"].clone(), first["height"].clone()),
        (json!(12), json!(8))
    );
    for field in DESCRIBE_FIELDS {
        assert_eq!(first[field], format!("first {field}"));
    }
    assert_eq!(first["failed"], false);
    assert_eq!(first["error"], Value::Null);

    let unreadable = &items[1];
    assert_eq!(unreadable["failed"], true);
    assert!(unreadable["error"]
        .as_str()
        .unwrap()
        .starts_with("The image could not be read"));
    assert_eq!(unreadable["width"], Value::Null);
    assert_eq!(unreadable["assetId"], Value::Null);
    for field in DESCRIBE_FIELDS {
        assert_eq!(
            unreadable[field], "",
            "a failed item still carries every field"
        );
    }

    let unparsable = &items[2];
    assert_eq!(unparsable["failed"], true);
    assert!(unparsable["error"]
        .as_str()
        .unwrap()
        .contains("not a JSON object"));
    assert_eq!(
        (unparsable["width"].clone(), unparsable["height"].clone()),
        (json!(6), json!(10)),
        "the size is known even when the description failed"
    );

    // Per-image progress: a "describing" post before each image and a running result after it.
    let messages: Vec<_> = posts(&api)
        .iter()
        .filter(|body| body["status"] == "running")
        .map(|body| body["message"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        messages,
        vec![
            "Describing image 1 of 3.",
            "Described 1 of 3 images.",
            "Describing image 2 of 3.",
            "Described 2 of 3 images.",
            "Describing image 3 of 3.",
            "Described 3 of 3 images.",
        ]
    );
    let running_counts: Vec<_> = posts(&api)
        .iter()
        .filter(|body| body["status"] == "running" && body["result"].is_object())
        .map(|body| body["result"]["items"].as_array().unwrap().len())
        .collect();
    assert_eq!(
        running_counts,
        vec![1, 2, 3],
        "results stream in as images finish"
    );
}

#[tokio::test]
async fn a_custom_prompt_and_a_generation_error_are_honoured_per_image() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let describer = StubDescriber::new(vec![Ok(Err(
        "Vision Captioner generation failed: boom".to_owned()
    ))]);
    // Re-describing ONE image is the same job with a one-item subset (the API filters the items).
    let mut subset = payload(&stage, Some("Describe only the clothing."));
    subset["items"] = json!([subset["items"][2].clone()]);
    run_job(&stage, subset, &describer)
        .await
        .expect("completes");
    let calls = describer.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert!(calls[0]
        .0
        .starts_with("Describe only the clothing.\n\n## OUTPUT"));
    let result = terminal(&api, "completed")["result"].clone();
    assert_eq!(result["promptSource"], "custom");
    assert_eq!(result["prompt"], "Describe only the clothing.");
    let items = result["items"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["itemId"], "item_3");
    assert_eq!(items[0]["failed"], true);
    assert_eq!(
        items[0]["error"],
        "Vision Captioner generation failed: boom"
    );
}

#[tokio::test]
async fn cancelling_mid_run_returns_the_images_already_described() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let mut describer = StubDescriber::new(vec![Ok(Ok(full_reply("first")))]);
    // The user cancels while the first image is being described.
    describer.cancel_after = Some((1, api.cancel.clone()));
    let error = run_job(&stage, payload(&stage, None), &describer)
        .await
        .expect_err("a canceled job ends canceled");
    assert!(matches!(error, WorkerError::Canceled(_)), "{error}");
    assert_eq!(describer.calls.lock().unwrap().len(), 1);
    assert!(
        posts(&api).iter().all(|body| body["status"] != "completed"),
        "a canceled job never reports completed"
    );
    let canceled = terminal(&api, "canceled");
    let result = &canceled["result"];
    assert_eq!(result["canceled"], true);
    assert_eq!(result["itemCount"], 3);
    assert_eq!(result["processedCount"], 1);
    let items = result["items"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["itemId"], "item_1");
    assert_eq!(items[0]["failed"], false);
    assert_eq!(items[0]["description"], "first description");
}

#[tokio::test]
async fn a_model_that_cannot_load_fails_the_job() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let describer = StubDescriber::new(vec![Err(WorkerError::Engine(
        "Vision Captioner load failed: missing weights".to_owned(),
    ))]);
    let error = run_job(&stage, payload(&stage, None), &describer)
        .await
        .expect_err("a load failure is not a per-image failure");
    assert!(error.to_string().contains("load failed"), "{error}");
    assert!(posts(&api).iter().all(|body| body["status"] != "completed"));
}

#[test]
fn the_request_refuses_other_models_unsafe_paths_and_oversized_prompts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let stage = Stage {
        settings: test_settings(dir.path()),
        dataset_root: dir.path().join("datasets").join("ds-1"),
        _dir: dir,
    };
    let base = payload(&stage, None);
    let refuse = |payload: Value, expect: &str| {
        let error = parse_batch_describe_request(
            &stage.settings,
            payload.as_object().expect("payload object"),
        )
        .err()
        .expect("refused");
        assert!(matches!(error, WorkerError::InvalidPayload(_)), "{error}");
        assert!(error.to_string().contains(expect), "{error}");
    };

    let mut other_model = base.clone();
    other_model["model"] = json!("someone/other-vlm");
    refuse(other_model, "does not support model");

    let mut outside = base.clone();
    outside["items"] = json!([{
        "itemId": "item_1",
        "imagePath": stage.dataset_root.join("../../escape.png").display().to_string(),
    }]);
    refuse(outside, "Batch describe item item_1 imagePath");

    let mut empty = base.clone();
    empty["items"] = json!([]);
    refuse(empty, "no images to describe");

    let mut long = base.clone();
    long["prompt"] =
        json!("x".repeat(sceneworks_core::dataset_batch_describe::MAX_DESCRIBE_PROMPT_CHARS + 1));
    refuse(long, "longer than");

    let mut not_text = base;
    not_text["prompt"] = json!(["a list"]);
    refuse(not_text, "must be a string");
}
