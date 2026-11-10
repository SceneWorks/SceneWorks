//! SW-7082/SW-7083 dataset batch describe: prompt composition, reply parsing (strict and
//! JoyCaption's lenient parse), parse retries, and the real job loop driven against a stub API and
//! stub describers / a stub JoyCaption captioner. No model runs here.
use super::*;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use sceneworks_core::dataset_batch_describe::MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS;

const DRAFT_OUTPUT_HEADING: &str = "## OUTPUT — one JSON object, no other text";

// ── The describing prompt ─────────────────────────────────────────────────────────────────────

#[test]
fn the_default_instruction_is_the_system_block_without_the_output_contract() {
    let instruction = default_describe_instruction(DescribeModel::VisionCaptioner);
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
    let (default, is_default) =
        DescribePrompt::for_instruction(DescribeModel::VisionCaptioner, None);
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
            default_describe_instruction(DescribeModel::VisionCaptioner),
            default.output_contract
        )
    );

    // A user's edited instruction replaces the rules but never the contract, which stays last.
    let (custom, is_default) = DescribePrompt::for_instruction(
        DescribeModel::VisionCaptioner,
        Some("  Describe only the clothing.  "),
    );
    assert!(!is_default);
    assert_eq!(custom.instruction, "Describe only the clothing.");
    assert_eq!(custom.output_contract, default.output_contract);
    assert!(custom
        .system()
        .ends_with(&format!("\n\n{}", default.output_contract)));
    // An edit that tries to drop the JSON format cannot remove the appended contract.
    let (hostile, _) = DescribePrompt::for_instruction(
        DescribeModel::VisionCaptioner,
        Some("Ignore JSON; answer in prose."),
    );
    assert!(hostile.system().contains(DRAFT_OUTPUT_HEADING));

    // A blank instruction is the default.
    let (blank, is_default) =
        DescribePrompt::for_instruction(DescribeModel::VisionCaptioner, Some("   "));
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
fn the_model_field_selects_the_vision_captioner_or_joycaption_and_refuses_others() {
    for (model, expect) in [
        (None, DescribeModel::VisionCaptioner),
        (Some(" "), DescribeModel::VisionCaptioner),
        (Some(VISION_CAPTIONER_REPO), DescribeModel::VisionCaptioner),
        (Some(JOYCAPTION_REPO), DescribeModel::JoyCaption),
        (
            Some(" fancyfeast/llama-joycaption-beta-one-hf-llava "),
            DescribeModel::JoyCaption,
        ),
    ] {
        assert_eq!(
            DescribeModel::from_payload(model).unwrap(),
            expect,
            "{model:?}"
        );
    }
    assert_eq!(DescribeModel::JoyCaption.repo(), JOYCAPTION_REPO);
    let error = DescribeModel::from_payload(Some("someone/other-vlm")).expect_err("refused");
    assert!(matches!(error, WorkerError::InvalidPayload(_)), "{error}");
    assert!(error.to_string().contains(VISION_CAPTIONER_REPO), "{error}");
    assert!(error.to_string().contains(JOYCAPTION_REPO), "{error}");
}

// ── JoyCaption's prompt and lenient parse ─────────────────────────────────────────────────────

/// The prompt-length ceiling of both JoyCaption providers at the pinned inference rev
/// (mlx-gen `caption/joycaption.rs` and candle-gen-joycaption `prompt.rs` `capabilities()`:
/// `max_prompt_chars: 4000`); a longer prompt is refused before generation.
const JOYCAPTION_PROVIDER_MAX_PROMPT_CHARS: usize = 4_000;

#[test]
fn the_joycaption_prompt_is_one_turn_that_fits_the_providers_limit() {
    let (prompt, is_default) = DescribePrompt::for_instruction(DescribeModel::JoyCaption, None);
    assert!(is_default);
    let instruction = default_describe_instruction(DescribeModel::JoyCaption);
    assert_eq!(prompt.instruction, instruction);
    assert_ne!(
        instruction,
        default_describe_instruction(DescribeModel::VisionCaptioner),
        "JoyCaption gets the condensed rules"
    );
    for rule in [
        "LABELS:",
        "man_N, woman_N, boy_N, girl_N",
        "NO PHYSICAL TRAITS:",
        "DESCRIPTION:",
        "ENVIRONMENT:",
        "STYLE:",
        "CAMERA:",
        "COMPOSITION:",
        "LIGHTING:",
    ] {
        assert!(
            instruction.contains(rule),
            "JoyCaption default keeps {rule:?}"
        );
    }
    assert!(!instruction.contains(DRAFT_OUTPUT_HEADING));
    // One prompt: instruction, then the contract, then the user line.
    let (vision, _) = DescribePrompt::for_instruction(DescribeModel::VisionCaptioner, None);
    assert_eq!(prompt.output_contract, vision.output_contract);
    assert_eq!(
        prompt.single_turn(),
        format!(
            "{instruction}\n\n{}\n\n{}",
            prompt.output_contract, prompt.user
        )
    );
    assert!(prompt.single_turn().chars().count() <= JOYCAPTION_PROVIDER_MAX_PROMPT_CHARS);
    // The longest instruction the API accepts for JoyCaption still fits the provider's limit.
    let longest = "x".repeat(MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS);
    let (custom, is_default) =
        DescribePrompt::for_instruction(DescribeModel::JoyCaption, Some(&longest));
    assert!(!is_default);
    assert!(
        custom.single_turn().chars().count() <= JOYCAPTION_PROVIDER_MAX_PROMPT_CHARS,
        "{} chars",
        custom.single_turn().chars().count()
    );
}

#[test]
fn the_lenient_parse_finds_the_object_inside_fences_and_prose() {
    let expected = parse_described_fields(&full_reply("img")).expect("strict parses");
    for raw in [
        full_reply("img"),
        format!("```json\n{}\n```", full_reply("img")),
        format!("Here is the JSON you asked for:\n{}", full_reply("img")),
        format!(
            "Sure! Here it is:\n```json\n{}\n```\nLet me know if you need {{more}} detail.",
            full_reply("img")
        ),
        format!("{{not json}} then {}", full_reply("img")),
        format!("{} {}", full_reply("img"), full_reply("second")),
        format!("<think>planning</think>{}", full_reply("img")),
    ] {
        assert_eq!(
            parse_lenient_described_fields(&raw).expect(&raw),
            expected,
            "{raw}"
        );
    }
    // Field names are matched ignoring case; a missing field is an empty string.
    let parsed = parse_lenient_described_fields(
        "{\"Description\": \"a man, man_1, waving\", \"LIGHTING\": \"soft window light\"}",
    )
    .expect("parses");
    assert_eq!(parsed.fields[0], "a man, man_1, waving");
    assert_eq!(parsed.fields[4], "soft window light");
    assert!(parsed.fields[1..4].iter().all(String::is_empty));
    assert!(parsed.fields[5].is_empty());
}

#[test]
fn the_lenient_parse_still_flags_unusable_replies() {
    for (raw, expect) in [
        ("A woman stands by a window.", "no complete JSON object"),
        ("[\"a\", \"b\"]", "no complete JSON object"),
        (
            "{\"description\": \"a woman, woman_1, cut off mid",
            "no complete JSON object",
        ),
        ("Here: {}", "none of the six"),
        ("{\"caption\": \"a woman\"}", "none of the six"),
        ("{\"camera\": 35}", "\"camera\" field is not a string"),
    ] {
        let error = parse_lenient_described_fields(raw).expect_err(raw);
        assert!(error.contains(expect), "{raw}: {error}");
    }
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

/// What the stub API saw (progress posts and `/result-items` appends), and the cancel flag it
/// reports.
#[derive(Clone, Default)]
struct StubApi {
    progress: Arc<Mutex<Vec<Value>>>,
    appends: Arc<Mutex<Vec<Value>>>,
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
    .route(
        "/api/v1/jobs/:job_id/result-items",
        axum::routing::post(
            |axum::extract::State(state): axum::extract::State<StubApi>,
             axum::Json(body): axum::Json<Value>| async move {
                let mut appends = state.appends.lock().expect("appends lock");
                appends.push(body);
                axum::Json(json!({ "total": appends.len() }))
            },
        ),
    )
    .with_state(state.clone());
    (crate::test_api_stub::serve(app).await, state)
}

/// The items the job appended to `/result-items`, in position order (a re-append of a position
/// replaces it, as the store does by item id).
fn stored_items(api: &StubApi) -> Vec<Value> {
    let mut by_position = std::collections::BTreeMap::new();
    for body in api.appends.lock().expect("appends lock").iter() {
        assert_eq!(body["workerId"], "test-worker", "appends name their worker");
        for entry in body["items"].as_array().expect("append items") {
            by_position.insert(
                entry["position"].as_u64().expect("position"),
                entry["item"].clone(),
            );
        }
    }
    by_position.into_values().collect()
}

/// One recorded describe call: (system message, user message, image size).
type DescribeCall = (String, String, (u32, u32));

/// A describer that records each call's prompt and image size, answers queued replies in order,
/// and optionally requests a job cancel once it has answered `cancel_after` images.
struct StubDescriber {
    calls: Mutex<Vec<DescribeCall>>,
    replies: Mutex<VecDeque<WorkerResult<Result<String, String>>>>,
    cancel_after: Option<(usize, Arc<AtomicBool>)>,
    /// `Some(retries)` behaves like JoyCaption: the lenient parse and that many parse retries.
    joycaption_like: Option<usize>,
}

impl StubDescriber {
    fn new(replies: Vec<WorkerResult<Result<String, String>>>) -> Self {
        Self {
            calls: Mutex::default(),
            replies: Mutex::new(replies.into()),
            cancel_after: None,
            joycaption_like: None,
        }
    }

    fn joycaption_like(replies: Vec<WorkerResult<Result<String, String>>>) -> Self {
        Self {
            joycaption_like: Some(JOYCAPTION_PARSE_RETRIES),
            ..Self::new(replies)
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

    fn parse(&self, raw: &str) -> Result<DescribedFields, String> {
        match self.joycaption_like {
            Some(_) => parse_lenient_described_fields(raw),
            None => parse_described_fields(raw),
        }
    }

    fn parse_retries(&self) -> usize {
        self.joycaption_like.unwrap_or(0)
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

async fn run_job(
    stage: &Stage,
    payload: Value,
    describer: &dyn BatchDescriber,
) -> WorkerResult<()> {
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
    let (default_prompt, _) = DescribePrompt::for_instruction(DescribeModel::VisionCaptioner, None);
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
    assert_eq!(
        result["prompt"],
        default_describe_instruction(DescribeModel::VisionCaptioner)
    );
    assert_eq!(result["itemCount"], 3);
    assert_eq!(result["processedCount"], 3);
    assert_eq!(result["describedCount"], 1);
    assert_eq!(result["failedCount"], 2);
    assert_eq!(result["canceled"], false);
    assert!(
        result.get("items").is_none(),
        "the items are stored per item, never in the progress result"
    );
    let items = stored_items(&api);
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
        .map(|body| body["result"]["processedCount"].clone())
        .collect();
    assert_eq!(
        running_counts,
        vec![json!(1), json!(2), json!(3)],
        "counts stream in as images finish"
    );
    let appended: Vec<_> = api
        .appends
        .lock()
        .unwrap()
        .iter()
        .map(|body| body["items"][0]["position"].clone())
        .collect();
    assert_eq!(
        appended,
        vec![json!(0), json!(1), json!(2)],
        "one append per finished image"
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
    let items = stored_items(&api);
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
    let items = stored_items(&api);
    assert_eq!(
        items.len(),
        1,
        "the canceled job keeps the image it finished"
    );
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

// ── Parse retries (SW-7083) ───────────────────────────────────────────────────────────────────

/// The payload with only item_1 (12x8), for `model`.
fn one_image_payload(stage: &Stage, model: &str) -> Value {
    let mut one = payload(stage, None);
    one["model"] = json!(model);
    one["items"] = json!([one["items"][0].clone()]);
    one
}

#[tokio::test]
async fn a_malformed_reply_is_described_again_until_it_parses() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let describer = StubDescriber::joycaption_like(vec![
        Ok(Ok("A woman, woman_1, stands by a window.".to_owned())),
        Ok(Ok(format!("Here you go: {}", full_reply("retry")))),
    ]);
    run_job(
        &stage,
        one_image_payload(&stage, JOYCAPTION_REPO),
        &describer,
    )
    .await
    .expect("completes");
    assert_eq!(describer.calls.lock().unwrap().len(), 2, "one retry");
    let completed = terminal(&api, "completed");
    assert_eq!(completed["message"], "Described 1 image(s).");
    let item = &stored_items(&api)[0];
    assert_eq!(item["failed"], false);
    assert_eq!(item["description"], "retry description");
    assert_eq!(completed["result"]["model"], JOYCAPTION_REPO);
    assert!(
        posts(&api)
            .iter()
            .any(|body| body["message"].as_str().is_some_and(
                |message| message.starts_with("Describing image 1 of 1 again (attempt 2)")
            )),
        "the retry is visible in progress"
    );
}

#[tokio::test]
async fn an_image_whose_replies_never_parse_is_flagged_after_the_last_retry() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let describer = StubDescriber::joycaption_like(vec![
        Ok(Ok("no json".to_owned())),
        Ok(Ok("{}".to_owned())),
        Ok(Ok("still no json".to_owned())),
    ]);
    run_job(
        &stage,
        one_image_payload(&stage, JOYCAPTION_REPO),
        &describer,
    )
    .await
    .expect("a flagged image does not fail the job");
    assert_eq!(
        describer.calls.lock().unwrap().len(),
        1 + JOYCAPTION_PARSE_RETRIES,
        "bounded retries"
    );
    let completed = terminal(&api, "completed");
    let result = &completed["result"];
    assert_eq!(result["failedCount"], 1);
    let items = stored_items(&api);
    assert_eq!(items.len(), 1, "the failed image is returned, not dropped");
    assert_eq!(items[0]["itemId"], "item_1");
    assert_eq!(items[0]["failed"], true);
    assert_eq!(
        (items[0]["width"].clone(), items[0]["height"].clone()),
        (json!(12), json!(8))
    );
    let error = items[0]["error"].as_str().unwrap();
    assert!(error.contains("no complete JSON object"), "{error}");
    assert!(error.contains("(after 3 attempts)"), "{error}");
}

#[tokio::test]
async fn a_generation_error_is_flagged_without_a_retry() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let describer = StubDescriber::joycaption_like(vec![Ok(Err(
        "JoyCaption MLX generation failed: boom".to_owned(),
    ))]);
    run_job(
        &stage,
        one_image_payload(&stage, JOYCAPTION_REPO),
        &describer,
    )
    .await
    .expect("completes");
    assert_eq!(describer.calls.lock().unwrap().len(), 1);
    assert_eq!(terminal(&api, "completed")["result"]["failedCount"], 1);
    let item = &stored_items(&api)[0];
    assert_eq!(item["failed"], true);
    assert_eq!(item["error"], "JoyCaption MLX generation failed: boom");
}

#[tokio::test]
async fn the_vision_captioner_is_not_retried() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let describer = StubDescriber::new(vec![Ok(Ok("no json".to_owned()))]);
    run_job(
        &stage,
        one_image_payload(&stage, VISION_CAPTIONER_REPO),
        &describer,
    )
    .await
    .expect("completes");
    assert_eq!(describer.calls.lock().unwrap().len(), 1);
    assert_eq!(terminal(&api, "completed")["result"]["failedCount"], 1);
}

#[tokio::test]
async fn a_cancel_before_a_retry_ends_the_job_without_another_attempt() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let mut describer = StubDescriber::joycaption_like(vec![Ok(Ok("no json".to_owned()))]);
    // The user cancels while the first attempt runs; its reply is malformed.
    describer.cancel_after = Some((1, api.cancel.clone()));
    let error = run_job(
        &stage,
        one_image_payload(&stage, JOYCAPTION_REPO),
        &describer,
    )
    .await
    .expect_err("canceled");
    assert!(matches!(error, WorkerError::Canceled(_)), "{error}");
    assert_eq!(
        describer.calls.lock().unwrap().len(),
        1,
        "no retry after cancel"
    );
    let canceled = terminal(&api, "canceled");
    assert_eq!(canceled["result"]["processedCount"], 0);
}

// ── The JoyCaption describer over a stub captioner ────────────────────────────────────────────

use gen_core::{
    CaptionCapabilities, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor,
};

/// What the stub captioner saw on one call.
#[derive(Clone, Debug)]
struct CaptionCall {
    prompt: String,
    custom_prompt: String,
    max_new_tokens: u32,
    temperature: f32,
    top_p: f32,
    size: (u32, u32),
    canceled_before: bool,
    canceled_after_progress: bool,
}

#[derive(Default)]
struct CaptionLog {
    loads: usize,
    calls: Vec<CaptionCall>,
}

/// A captioner answering queued replies. When `trip` is set, it trips that job cancel flag mid-call
/// (before its progress event), as a user cancel during decode would.
struct StubCaptioner {
    descriptor: CaptionerDescriptor,
    log: Arc<Mutex<CaptionLog>>,
    replies: Mutex<VecDeque<String>>,
    trip: Option<CancelFlag>,
}

impl Captioner for StubCaptioner {
    fn descriptor(&self) -> &CaptionerDescriptor {
        &self.descriptor
    }
    fn validate(&self, _req: &CaptionRequest) -> gen_core::Result<()> {
        Ok(())
    }
    fn caption(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(gen_core::Progress),
    ) -> gen_core::Result<CaptionOutput> {
        let canceled_before = req.cancel.is_cancelled();
        if let Some(flag) = &self.trip {
            flag.cancel();
        }
        on_progress(gen_core::Progress::Step {
            current: 1,
            total: 1,
        });
        self.log.lock().unwrap().calls.push(CaptionCall {
            prompt: req.prompt.clone(),
            custom_prompt: req.options.custom_prompt.clone(),
            max_new_tokens: req.sampling.max_new_tokens,
            temperature: req.sampling.temperature,
            top_p: req.sampling.top_p,
            size: (req.image.width, req.image.height),
            canceled_before,
            canceled_after_progress: req.cancel.is_cancelled(),
        });
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        Ok(CaptionOutput {
            text: self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("a queued caption"),
            generated_tokens: None,
            finish_reason: None,
        })
    }
}

/// A JoyCaption describer whose loader yields a stub captioner (or fails when `replies` is None).
fn stub_joycaption(
    replies: Option<Vec<String>>,
    trip: Option<CancelFlag>,
) -> (JoyCaptionDescriber, Arc<Mutex<CaptionLog>>) {
    let log = Arc::new(Mutex::new(CaptionLog::default()));
    let loader_log = log.clone();
    let describer = JoyCaptionDescriber::spawn(
        PathBuf::from("/models/joycaption"),
        "describe-job".to_owned(),
        "mlx".to_owned(),
        Box::new(move |id, _spec| {
            assert_eq!(id, JOYCAPTION_REPO);
            loader_log.lock().unwrap().loads += 1;
            let Some(replies) = replies else {
                return Err(gen_core::Error::Msg("missing weights".to_owned()));
            };
            Ok(Box::new(StubCaptioner {
                descriptor: CaptionerDescriptor {
                    id: "fancyfeast/llama-joycaption-beta-one-hf-llava",
                    family: "joycaption",
                    backend: "stub",
                    capabilities: CaptionCapabilities::default(),
                },
                log: loader_log,
                replies: Mutex::new(replies.into()),
                trip,
            }) as Box<dyn Captioner>)
        }),
    );
    (describer, log)
}

#[tokio::test]
async fn joycaption_describes_every_image_with_one_loaded_captioner_and_its_one_turn_prompt() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let (describer, log) = stub_joycaption(
        Some(vec![
            format!("```json\n{}\n```", full_reply("first")),
            format!("The image shows: {}", full_reply("third")),
        ]),
        None,
    );
    let mut job_payload = payload(&stage, None);
    job_payload["model"] = json!(JOYCAPTION_REPO);
    run_job(&stage, job_payload, &describer)
        .await
        .expect("completes");

    let log = log.lock().unwrap();
    assert_eq!(log.loads, 1, "the model loads once for the dataset");
    assert_eq!(log.calls.len(), 2, "item_2 has no file");
    let (prompt, _) = DescribePrompt::for_instruction(DescribeModel::JoyCaption, None);
    for call in &log.calls {
        assert_eq!(call.prompt, prompt.single_turn());
        assert_eq!(
            call.custom_prompt, call.prompt,
            "custom_prompt is the prompt's source"
        );
        assert_eq!(call.max_new_tokens, 1024);
        assert_eq!((call.temperature, call.top_p), (0.6, 0.9));
    }
    assert_eq!(log.calls[0].size, (12, 8));
    assert_eq!(log.calls[1].size, (6, 10));

    let result = terminal(&api, "completed")["result"].clone();
    assert_eq!(result["model"], JOYCAPTION_REPO);
    assert_eq!(
        result["prompt"],
        default_describe_instruction(DescribeModel::JoyCaption)
    );
    let items = stored_items(&api);
    assert_eq!(items.len(), 3);
    for field in DESCRIBE_FIELDS {
        assert_eq!(items[0][field], format!("first {field}"));
        assert_eq!(items[2][field], format!("third {field}"));
    }
    assert_eq!(items[1]["failed"], true);
}

#[tokio::test]
async fn a_joycaption_model_that_cannot_load_fails_the_job() {
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let (describer, log) = stub_joycaption(None, None);
    let error = run_job(
        &stage,
        one_image_payload(&stage, JOYCAPTION_REPO),
        &describer,
    )
    .await
    .expect_err("a load failure fails the job");
    assert!(matches!(error, WorkerError::Engine(_)), "{error}");
    assert!(
        error
            .to_string()
            .contains("JoyCaption MLX load failed: missing weights"),
        "{error}"
    );
    assert_eq!(log.lock().unwrap().loads, 1);
    assert!(posts(&api).iter().all(|body| body["status"] != "completed"));
}

#[tokio::test]
async fn the_job_cancel_reaches_the_captioner_before_and_during_a_call() {
    // Already canceled when the call starts: the captioner's own flag is tripped up front.
    let (describer, log) = stub_joycaption(Some(vec![]), None);
    let cancel = CancelFlag::new();
    cancel.cancel();
    let image = image::DynamicImage::new_rgb8(4, 4);
    let (prompt, _) = DescribePrompt::for_instruction(DescribeModel::JoyCaption, None);
    let reply = describer
        .describe(image.clone(), &prompt, cancel)
        .await
        .expect("no load error");
    assert!(reply.expect_err("canceled").contains("canceled"));
    assert!(log.lock().unwrap().calls[0].canceled_before);

    // Canceled mid-decode: the next progress event bridges it.
    let cancel = CancelFlag::new();
    let (describer, log) = stub_joycaption(Some(vec![]), Some(cancel.clone()));
    let reply = describer
        .describe(image, &prompt, cancel)
        .await
        .expect("no load error");
    assert!(reply.is_err());
    let call = log.lock().unwrap().calls[0].clone();
    assert!(!call.canceled_before);
    assert!(
        call.canceled_after_progress,
        "the progress callback bridges the cancel"
    );
}

// ── Bounded posts (SW-7082 review) ────────────────────────────────────────────────────────────

/// A large dataset of long descriptions: every progress and terminal post stays the same small
/// size however many images are done, each append carries one bounded item, and the stored items
/// come back complete and in dataset order.
#[tokio::test]
async fn posts_stay_bounded_for_a_large_dataset_and_every_item_is_stored_in_order() {
    const IMAGES: usize = 150;
    let (url, api) = spawn_stub_api().await;
    let stage = stage(url);
    let image_path = stage.dataset_root.join("images/item_1.png");
    let mut job_payload = payload(&stage, None);
    job_payload["items"] = Value::Array(
        (0..IMAGES)
            .map(|index| {
                json!({
                    "itemId": format!("item_{index}"),
                    "imagePath": image_path.display().to_string(),
                    "relativePath": format!("images/item_{index}.png"),
                    "assetId": Value::Null,
                })
            })
            .collect(),
    );
    // ~6 KB of text per image: 150 images would be ~900 KB in a cumulative result.
    let long = "word ".repeat(200);
    let describer = StubDescriber::new(
        (0..IMAGES)
            .map(|index| {
                Ok(Ok(reply(
                    &DESCRIBE_FIELDS
                        .iter()
                        .map(|field| (*field, json!(format!("{index} {field} {long}"))))
                        .collect::<Vec<_>>(),
                )))
            })
            .collect(),
    );
    run_job(&stage, job_payload, &describer)
        .await
        .expect("completes");

    // Every post that carries a result carries the same echo (the instruction text included) and
    // counts, so its size is flat across the run instead of growing ~6 KB per image.
    let sizes: Vec<usize> = posts(&api)
        .iter()
        .filter(|body| body["result"].is_object())
        .map(|body| body.to_string().len())
        .collect();
    assert_eq!(
        sizes.len(),
        IMAGES + 1,
        "a running post per image and the terminal post"
    );
    let (smallest, largest) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
    assert!(
        largest - smallest < 256,
        "result posts range {smallest}..{largest} bytes; they must not grow with the dataset"
    );
    assert!(largest < 16 * 1024, "{largest} bytes");
    let completed = terminal(&api, "completed");
    assert_eq!(completed["result"]["processedCount"], IMAGES);
    assert!(completed["result"].get("items").is_none());

    let appends = api.appends.lock().unwrap().clone();
    assert_eq!(appends.len(), IMAGES, "one append per image");
    for body in &appends {
        assert_eq!(body["items"].as_array().unwrap().len(), 1);
        assert!(
            body.to_string().len() <= sceneworks_core::jobs_store::MAX_JOB_RESULT_ITEM_BYTES,
            "an append carries one bounded item"
        );
    }
    let items = stored_items(&api);
    let ids: Vec<String> = items
        .iter()
        .map(|item| item["itemId"].as_str().unwrap().to_owned())
        .collect();
    let expected: Vec<String> = (0..IMAGES).map(|index| format!("item_{index}")).collect();
    assert_eq!(ids, expected);
    assert!(items[IMAGES - 1]["description"]
        .as_str()
        .unwrap()
        .starts_with(&format!("{} description word", IMAGES - 1)));
}

/// The worst case one image can produce — every field at its cap, every character one that JSON
/// escapes to six bytes, the longest error — still fits the store's per-item limit, and a field
/// past the cap is cut.
#[test]
fn one_image_can_never_exceed_the_per_item_limit() {
    let escaped = "\u{1}".repeat(MAX_DESCRIBE_FIELD_CHARS * 2);
    let raw = reply(
        &DESCRIBE_FIELDS
            .iter()
            .map(|field| (*field, json!(escaped)))
            .collect::<Vec<_>>(),
    );
    let parsed = parse_described_fields(&raw).expect("parses");
    for field in &parsed.fields {
        assert_eq!(
            field.chars().count(),
            MAX_DESCRIBE_FIELD_CHARS + 1,
            "cut, with an ellipsis"
        );
    }
    let item = DescribeItem {
        item_id: "item_1".to_owned(),
        image_path: PathBuf::from("/data/images/item_1.png"),
        relative_path: "images/item_1.png".to_owned(),
        asset_id: Some("asset-1".to_owned()),
    };
    let described = DescribedItem {
        item: item.clone(),
        size: Some((4096, 4096)),
        outcome: Ok(parsed),
    };
    let encoded = stored_item_json(&described).to_string().len();
    assert!(
        encoded <= sceneworks_core::jobs_store::MAX_JOB_RESULT_ITEM_BYTES,
        "{encoded} bytes"
    );
    let failed = DescribedItem {
        item,
        size: None,
        outcome: Err("\u{1}".repeat(50_000)),
    };
    assert!(
        stored_item_json(&failed).to_string().len()
            <= sceneworks_core::jobs_store::MAX_JOB_RESULT_ITEM_BYTES
    );
}

/// Should an item still exceed the limit (here through an absurd path), it is stored flagged and
/// without its text rather than refused, so the job never fails on the append.
#[test]
fn an_item_over_the_limit_is_stored_flagged_without_its_text() {
    let described = DescribedItem {
        item: DescribeItem {
            item_id: "item_1".to_owned(),
            image_path: PathBuf::from("/data/x.png"),
            relative_path: "\u{1}".repeat(sceneworks_core::jobs_store::MAX_JOB_RESULT_ITEM_BYTES),
            asset_id: None,
        },
        size: Some((10, 20)),
        outcome: Ok(DescribedFields {
            fields: std::array::from_fn(|_| "text".to_owned()),
        }),
    };
    let stored = stored_item_json(&described);
    assert!(stored.to_string().len() <= sceneworks_core::jobs_store::MAX_JOB_RESULT_ITEM_BYTES);
    assert_eq!(stored["itemId"], "item_1");
    assert_eq!(stored["failed"], true);
    assert_eq!(stored["width"], 10);
    assert_eq!(stored["description"], "");
}
