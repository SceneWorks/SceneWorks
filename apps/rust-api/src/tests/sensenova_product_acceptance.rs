//! SW-7189 dedicated API-to-native-worker proof. All learned execution enters the ignored parent test.
use super::*;

const MODEL: &str = "sensenova/SenseNova-Loop-9B";
const MODEL_ID: &str = "vision_sensenova_loop_9b";
const FIELDS: [&str; 7] = [
    "description",
    "pose",
    "style",
    "camera",
    "composition",
    "lighting",
    "environment",
];
const SCENES: [(&str, &str, [u8; 3]); 2] = [
    ("circle.png", "red", [218, 20, 30]),
    ("square.png", "blue", [20, 40, 220]),
];

const PUBLISHER: &str =
    include_str!("../../../../scripts/fixtures/sensenova-loop-9b-publisher.json");
use sceneworks_core::model_artifacts::portable_binding::{
    resolve_pinned_hf_files, PortableArtifactFile,
};
use std::collections::BTreeMap;

fn official_files(
    repo: &str,
    rev: &str,
    selectors: &[String],
) -> Result<Vec<PortableArtifactFile>> {
    let publisher: Value = serde_json::from_str(PUBLISHER).map_err(|e| e.to_string())?;
    require(
        publisher["repository"] == repo && publisher["revision"] == rev,
        "incorrect BF16 publisher",
    )?;
    let trusted = publisher["files"]
        .as_object()
        .ok_or("invalid BF16 ledger")?;
    require(
        trusted.len() == selectors.len()
            && selectors
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == selectors.len(),
        "incomplete BF16 native file list",
    )?;
    selectors
        .iter()
        .map(|name| {
            let source = trusted
                .get(name)
                .ok_or("BF16 selector absent from publisher ledger")?;
            Ok(PortableArtifactFile {
                relative_path: name.into(),
                size_bytes: source["bytes"].as_u64().ok_or("missing file size")?,
                sha256: format!("sha256:{}", field(source, "sha256")?),
            })
        })
        .collect::<Result<Vec<_>>>()
}

/// Fetch public quantized publisher metadata before creating the offline app. The gated BF16
/// manifest was captured from the official pinned revision with authorized access.
pub(super) async fn publisher_metadata(
    tier: &str,
    root: &Path,
) -> Result<BTreeMap<(String, String), Vec<PortableArtifactFile>>> {
    let entry = model_entry(MODEL_ID)?;
    let mut ledger = BTreeMap::new();
    let client = reqwest::Client::new();
    for row in entry["downloads"]
        .as_array()
        .ok_or("missing downloads")?
        .iter()
        .filter(|row| {
            row["variant"] == tier
                && row["platforms"]
                    .as_array()
                    .is_none_or(|platforms| platforms.iter().any(|os| os == std::env::consts::OS))
                && row["pendingArtifact"] != true
                && row["revision"] != "0000000000000000000000000000000000000000"
        })
    {
        let repo = field(row, "repo")?;
        let rev = field(row, "revision")?;
        let selectors: Vec<String> =
            serde_json::from_value(row["files"].clone()).map_err(|e| e.to_string())?;
        let files = if tier == "bf16" {
            official_files(repo, rev, &selectors)?
        } else {
            resolve_pinned_hf_files(
                &client,
                "https://huggingface.co",
                None,
                repo,
                rev,
                &selectors,
            )
            .await
            .map_err(|e| e.to_string())?
        };
        require(
            !files.is_empty() && files.len() == selectors.len(),
            "publisher did not authenticate every selected file",
        )?;
        ledger.insert((repo.to_owned(), rev.to_owned()), files);
    }
    require(!ledger.is_empty(), "no published tier")?;
    let evidence: Vec<_> = ledger
        .iter()
        .map(|((repo, rev), files)| json!({"repo":repo,"revision":rev,"files":files}))
        .collect();
    save(root, "publisher-metadata.json", &json!(evidence))?;
    Ok(ledger)
}

pub(super) fn configure_metadata(
    state: &crate::AppState,
    ledger: BTreeMap<(String, String), Vec<PortableArtifactFile>>,
) {
    *state.test_hooks.pinned_metadata_remote.lock() =
        Some(Arc::new(move |repo, rev, selectors| {
            let entries = ledger
                .get(&(repo.into(), rev.into()))
                .ok_or_else(|| format!("unknown publisher metadata: {repo}@{rev}"))?;
            selectors
                .iter()
                .map(|name| {
                    entries
                        .iter()
                        .find(|entry| entry.relative_path == Path::new(name))
                        .cloned()
                        .ok_or_else(|| format!("publisher metadata absent: {repo}@{rev}/{name}"))
                })
                .collect()
        }));
}

fn grounded(text: &str, index: usize) -> bool {
    let text = text.to_lowercase();
    let words: Vec<_> = text.split(|c: char| !c.is_alphanumeric()).collect();
    let shapes: &[&str] = if index == 0 {
        &["circle", "circular", "disk", "disc"]
    } else {
        &["square", "quadrilateral"]
    };
    words.contains(&SCENES[index].1) && shapes.iter().any(|shape| words.contains(shape))
}
fn scene_png(index: usize) -> Result<Vec<u8>> {
    let image = image::RgbImage::from_fn(384, 256, |x, y| {
        let dx = x.abs_diff(192);
        let dy = y.abs_diff(128);
        let foreground = if index == 0 {
            dx * dx + dy * dy < 96 * 96
        } else {
            dx < 96 && dy < 96
        };
        image::Rgb(if foreground {
            SCENES[index].2
        } else {
            [245, 245, 245]
        })
    });
    let mut png = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut png, image::ImageFormat::Png)
        .map_err(|error| error.to_string())?;
    Ok(png.into_inner())
}

fn check_binding(
    route: &sceneworks_core::model_artifacts::portable_binding::PortableRouteBinding,
    tier: &str,
) -> Result<()> {
    use sceneworks_core::model_artifacts::portable_binding::PortableSourceIdentity;
    use std::collections::BTreeSet;
    let backend = if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "candle"
    };
    require(
        route.backend == backend && route.platform == std::env::consts::OS,
        "unexpected worker route",
    )?;
    require(
        route.models.len() == 1 && route.models[0].model_id == MODEL_ID,
        "worker route selected a different model",
    )?;
    let [member] = route.models[0].members.as_slice() else {
        return Err("expected one bound tier artifact".into());
    };
    let PortableSourceIdentity::Revision { identity } = &member.source else {
        return Err("worker route omitted pinned revision".into());
    };
    let entry = model_entry(MODEL_ID)?;
    let rows = entry["downloads"]
        .as_array()
        .ok_or("missing download rows")?;
    let row = rows
        .iter()
        .find(|row| {
            row["variant"] == tier
                && row["platforms"].as_array().is_none_or(|platforms| {
                    platforms.iter().any(|platform| platform == &route.platform)
                })
        })
        .ok_or("missing selected tier on this platform")?;
    require(
        row["repo"] == identity.repository
            && row["revision"] == identity.revision
            && identity.variant == tier,
        "claimed model revision or tier differs from shipped selection",
    )?;
    let expected: BTreeSet<_> = row["files"]
        .as_array()
        .ok_or("missing manifest files")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let observed: BTreeSet<_> = member
        .files
        .iter()
        .map(|file| file.relative_path.to_string_lossy().replace('\\', "/"))
        .collect();
    require(
        !expected.is_empty()
            && observed.iter().map(String::as_str).collect::<BTreeSet<_>>() == expected,
        "claimed closure differs from enumerated tier files",
    )?;
    Ok(())
}

async fn execute(
    api: &Api,
    worker: &sceneworks_worker::Settings,
    job: &Value,
    status: &str,
    root: &Path,
    name: &str,
) -> Result<Value> {
    let id = field(job, "id")?;
    let model_free = job["payload"]["mode"] == "triggerOnly";
    native::run_bound_acceptance_job(worker, id, |route| {
        if model_free {
            require(
                route.is_none() && job["payload"].get("variant").is_none(),
                "trigger-only job selected model weights",
            )?;
        } else {
            check_binding(
                route.ok_or("model-bearing job has no binding")?,
                field(&job["payload"], "variant")?,
            )?;
        }
        save(
            root,
            &format!("{name}-binding.json"),
            &serde_json::to_value(route).map_err(|e| e.to_string())?,
        )
    })
    .await
    .map_err(|e| e.to_string())?;
    let completed = api
        .json("GET", &format!("/api/v1/jobs/{id}"), Value::Null)
        .await?;
    let metrics = api
        .json("GET", &format!("/api/v1/jobs/{id}/metrics"), Value::Null)
        .await?;
    save(
        root,
        &format!("{name}.json"),
        &json!({"job": completed, "metrics":metrics}),
    )?;
    require(
        completed["status"] == status,
        format!("{name}: expected {status}, got {}", completed["message"]),
    )?;
    let backend = if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "cuda"
    };
    require(
        metrics["backend"] == backend,
        format!("{name}: wrong metrics backend: {metrics}"),
    )?;
    require(
        metrics["totalMs"].as_u64().unwrap_or(0) > 0,
        format!("{name}: missing duration"),
    )?;
    // First native job from a fresh worker, same inputs across tiers. MLX's allocation high-water
    // mark and CUDA's sampled board memory are different measurement sources; retain both identities.
    if name == "batch" {
        require(
            metrics["peakMemoryBytes"].as_u64().unwrap_or(0) > 0,
            "missing measured batch memory peak",
        )?;
    }
    Ok(completed)
}

fn check_rows(page: &Value, dataset: &Value, expected: usize) -> Result<()> {
    let rows = page["items"]
        .as_array()
        .ok_or("missing persisted result items")?;
    require(
        rows.len() == expected && page["total"] == expected && page["nextOffset"].is_null(),
        "incomplete result page",
    )?;
    for (i, row) in rows.iter().enumerate() {
        require(
            row["itemId"] == dataset["items"][i]["id"]
                && row["width"] == 384
                && row["height"] == 256,
            "image order or dimensions changed",
        )?;
        require(
            row["failed"] == false && row["error"].is_null() && row["notice"].is_null(),
            "failed or truncated description",
        )?;
        for key in FIELDS {
            let text = row[key]
                .as_str()
                .ok_or_else(|| format!("missing string field {key}"))?;
            if !matches!(key, "pose" | "camera") {
                require(
                    !text.trim().is_empty(),
                    format!("empty description field {key}"),
                )?;
            }
        }
        require(
            grounded(
                &FIELDS
                    .iter()
                    .filter_map(|key| row[key].as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                i,
            ),
            "description lost the fixture's dominant color",
        )?;
    }
    Ok(())
}

/// Enqueue on a separate install with an empty private hub. Its catalog cache and missing
/// files cannot affect the real install, even when every supported tier is provisioned.
async fn verify_missing_tier(tier: &str, root: &Path) -> Result<()> {
    let temp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let mut settings = test_settings(&temp);
    settings.host_os = std::env::consts::OS.into();
    let _empty = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
        settings.data_dir.clone(),
        temp.path().join("empty-hub"),
    );
    super::super::moss_training_acceptance::seed_acceptance_manifests(&settings)?;
    let app = create_app(settings).map_err(|e| e.to_string())?;
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"Empty install"}),
    )
    .await;
    let training = format!("/api/v1/projects/{}/training", field(&project, "id")?);
    let (_, upload) = request_multipart_upload(
        app.clone(),
        &format!("{training}/uploads"),
        "circle.png",
        "image/png",
        &scene_png(0)?,
    )
    .await;
    let (_,dataset) = request(app.clone(),"POST",&format!("{training}/datasets"),json!({"name":"Empty install","items":[{"path":field(&upload["file"],"path")?,"displayName":"circle.png"}]})).await;
    let route = format!(
        "{training}/datasets/{}/batch-describe-jobs",
        field(&dataset, "id")?
    );
    let (status, body) = request(
        app.clone(),
        "POST",
        &route,
        json!({"model":MODEL,"variant":tier}),
    )
    .await;
    require(
        status == StatusCode::CONFLICT
            && body["code"] == "model_artifact_incomplete"
            && field(&body, "detail")?.contains(tier),
        format!("missing selected tier did not fail at enqueue: {status} {body}"),
    )?;
    let (_, jobs) = request(app, "GET", &route, Value::Null).await;
    require(jobs == json!([]), "missing tier enqueued a job")?;
    save(root, "missing-tier-refusal.json", &body)
}

async fn save_review(
    url: &str,
    root: &Path,
    review: &Value,
    dataset: &Value,
    tier: &str,
) -> Result<()> {
    save(
        root,
        "review-input.json",
        &json!({"items":review,"dataset":dataset,"model":MODEL,"variant":tier}),
    )?;
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/sensenova-review-batch.mjs");
    let export = tokio::process::Command::new("node")
        .arg(script)
        .arg(url)
        .arg(root)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    require(
        export.status.success(),
        format!(
            "review/save/export failed: {}",
            String::from_utf8_lossy(&export.stderr)
        ),
    )
}
pub(super) async fn exercise(
    api: &Api,
    worker: &sceneworks_worker::Settings,
    ledger: &HttpLedger,
    root: &Path,
    project: &str,
    tier: &str,
    receipt: &mut Value,
) -> Result<()> {
    let entry = model_entry(MODEL_ID)?;
    receipt["catalogId"] = json!(MODEL_ID);
    receipt["variant"] = json!(tier);
    receipt["compiledManifestSha256"] =
        json!(sha(&serde_json::to_vec(&entry).map_err(|e| e.to_string())?));
    let project_route = format!("/api/v1/projects/{project}");
    let mut items = Vec::new();
    let mut asset_id = String::new();
    for (i, (filename, _, _)) in SCENES.iter().enumerate() {
        let bytes = scene_png(i)?;
        std::fs::write(root.join(filename), &bytes).map_err(|e| e.to_string())?;
        let uploaded = api
            .upload(
                &format!("{project_route}/training/uploads"),
                filename,
                "image/png",
                &bytes,
            )
            .await?;
        items.push(json!({"path":field(&uploaded["file"], "path")?, "displayName":filename, "width":384, "height":256}));
        if i == 0 {
            let asset = api
                .upload(
                    &format!("{project_route}/assets"),
                    filename,
                    "image/png",
                    &bytes,
                )
                .await?;
            asset_id = field(&asset, "id")?.to_owned();
        }
    }
    let datasets = format!("{project_route}/training/datasets");
    let dataset = api
        .json(
            "POST",
            &datasets,
            json!({"name":"SenseNova product acceptance","modality":"image","items":items}),
        )
        .await?;
    let dataset_route = format!("{datasets}/{}", field(&dataset, "id")?);
    let batch = format!("{dataset_route}/batch-describe-jobs");
    let select = json!({"model":MODEL,"variant":tier});
    verify_missing_tier(tier, root).await?;
    let catalog = api.json("GET", "/api/v1/models", Value::Null).await?;
    let selected = catalog
        .as_array()
        .and_then(|rows| rows.iter().find(|model| model["id"] == MODEL_ID))
        .ok_or("model missing from API catalog")?;
    require(
        selected["variants"].as_array().is_some_and(|rows| {
            rows.iter()
                .any(|row| row["variant"] == tier && row["installState"] == "installed")
        }),
        "catalog did not project selected tier as installed",
    )?;
    save(root, "installed-tier.json", selected)?;
    let job = api.json("POST", &batch, select.clone()).await?;
    let run = execute(api, worker, &job, "completed", root, "batch").await?;
    let result = &run["result"];
    require(
        result["model"] == MODEL
            && result["variant"] == tier
            && result["describedCount"] == 2
            && result["failedCount"] == 0,
        "wrong batch model, tier or counts",
    )?;
    let page = api
        .json(
            "GET",
            &format!("/api/v1/jobs/{}/result-items", field(&job, "id")?),
            Value::Null,
        )
        .await?;
    check_rows(&page, &dataset, 2)?;
    save(root, "batch-result-items.json", &page)?;
    // Full product proof at q4 or the explicitly selected official BF16 profile. The tier probes reuse
    // the identical cold batch workload for tier qualification and comparable peak measurements.
    if tier != "q4" && receipt["case"] != "sensenova_bf16_product" {
        return Ok(());
    }

    let job = api.json("POST", &batch, select).await?;
    let id = field(&job, "id")?;
    *ledger.cancel_after_post.lock().unwrap() = Some(CancelOnPost {
        route: format!("/api/v1/jobs/{id}/result-items"),
        message: None,
        cancel_url: format!("{}/api/v1/jobs/{id}/cancel", api.url),
    });
    let canceled = execute(api, worker, &job, "canceled", root, "batch-canceled").await?;
    require(
        canceled["result"]["canceled"] == true && canceled["result"]["processedCount"] == 1,
        "partial batch cancel lost its counts",
    )?;
    let history = api.json("GET", &batch, Value::Null).await?;
    require(
        history[0]["id"] == id
            && history[0]["status"] == "canceled"
            && history[1]["id"] == run["id"],
        "dataset history lost the canceled or completed run",
    )?;
    let partial = api
        .json(
            "GET",
            &format!("/api/v1/jobs/{id}/result-items"),
            Value::Null,
        )
        .await?;
    check_rows(&partial, &dataset, 1)?;

    // A corrupted owned upload must produce one failed row and leave the job reviewable.
    // Restore its exact content before re-describing that image with Fit to.
    let retry_body = json!({"model":MODEL,"variant":tier,"itemIds":[dataset["items"][1]["id"]]});
    let broken = api.json("POST", &batch, retry_body.clone()).await?;
    let image = Path::new(field(&broken["payload"]["items"][0], "imagePath")?)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    require(
        image.starts_with(worker.data_dir.canonicalize().map_err(|e| e.to_string())?),
        "fixture outside isolated application data",
    )?;
    let original = std::fs::read(&image).map_err(|e| e.to_string())?;
    std::fs::write(&image, b"broken acceptance PNG").map_err(|e| e.to_string())?;
    let failed = execute(
        api,
        worker,
        &broken,
        "completed",
        root,
        "batch-broken-image",
    )
    .await;
    std::fs::write(&image, original).map_err(|e| e.to_string())?;
    let failed = failed?;
    require(
        failed["result"]["failedCount"] == 1,
        "missing per-image failure count",
    )?;
    let errors = api
        .json(
            "GET",
            &format!("/api/v1/jobs/{}/result-items", field(&failed, "id")?),
            Value::Null,
        )
        .await?;
    require(
        errors["total"] == 1
            && errors["items"][0]["itemId"] == dataset["items"][1]["id"]
            && errors["items"][0]["failed"] == true
            && field(&errors["items"][0], "error").is_ok(),
        "failed image did not retain its own error",
    )?;
    let mut retry_body = retry_body;
    retry_body["fitTo"] = json!("krea_2_turbo");
    let retry = api.json("POST", &batch, retry_body).await?;
    let recovered = execute(api, worker, &retry, "completed", root, "batch-fit-retry").await?;
    let recovered_rows = api
        .json(
            "GET",
            &format!("/api/v1/jobs/{}/result-items", field(&recovered, "id")?),
            Value::Null,
        )
        .await?;
    let row = &recovered_rows["items"][0];
    let condensed = &row["condensed"];
    require(
        recovered_rows["total"] == 1
            && row["itemId"] == dataset["items"][1]["id"]
            && row["failed"] == false
            && condensed["target"] == "krea_2_turbo"
            && condensed["failed"] == false
            && condensed["over"] == false
            && grounded(field(condensed, "text")?, 1),
        "retry did not retain a successful fitted prompt for the selected image",
    )?;
    require(
        condensed["tokens"]
            .as_u64()
            .is_some_and(|n| n > 0 && n <= condensed["limitTokens"].as_u64().unwrap_or(0)),
        "invalid fitted prompt token budget",
    )?;
    save(root, "fit-retry-result.json", &recovered_rows)?;
    let mut review = page["items"].clone();
    review[1] = row.clone();
    save_review(&api.url, root, &review, &dataset, tier).await?;

    let job = api
        .json(
            "POST",
            &format!("{dataset_route}/caption-jobs"),
            json!({"captioner":"sensenova_loop","modelNameOrPath":MODEL,"variant":tier,"triggerWords":["loopshape"]}),
        )
        .await?;
    let training = execute(api, worker, &job, "completed", root, "training-caption").await?;
    let result = &training["result"];
    require(
        result["captioner"] == "sensenova_loop"
            && result["modelNameOrPath"] == MODEL
            && result["variant"] == tier
            && result["captionedItemCount"] == 2,
        "wrong training caption identity, tier or count",
    )?;
    let updated = api.json("GET", &dataset_route, Value::Null).await?;
    for (i, _) in SCENES.iter().enumerate() {
        let item = &updated["items"][i];
        require(
            item["id"] == dataset["items"][i]["id"]
                && item["caption"]["source"] == "auto"
                && field(&item["caption"], "text")?.starts_with("loopshape")
                && grounded(field(&item["caption"], "text")?, i),
            "missing grounded auto-caption on its source image",
        )?;
    }

    let caption_route = format!("{dataset_route}/caption-jobs");
    let progress_route = format!("/api/v1/jobs/{}/progress", field(&training, "id")?);
    let updates: Vec<Value> = ledger
        .rows
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r["route"] == progress_route && r["method"] == "POST" && r["httpStatus"] == 200)
        .map(|r| r["request"].clone())
        .collect();
    let mut cursor = updates.iter();
    for stage in ["preparing", "running", "saving", "completed"] {
        let update = cursor
            .find(|r| r["stage"] == stage)
            .ok_or_else(|| format!("missing or out-of-order training progress stage {stage}"))?;
        let progress = update["progress"]
            .as_f64()
            .ok_or("missing numeric progress")?;
        require(
            if stage == "completed" {
                progress == 1.0
            } else {
                (0.0..1.0).contains(&progress)
            },
            format!("invalid training progress {progress} for {stage}"),
        )?;
    }
    save(root, "caption-progress.json", &json!(updates))?;
    let body = |mode: &str| json!({"captioner":"sensenova_loop","modelNameOrPath":MODEL,"variant":tier,"itemIds":[dataset["items"][0]["id"]],"mode":mode,"triggerWords":["loopshape"]});
    let subject_job = api
        .json("POST", &caption_route, body("subjectOnly"))
        .await?;
    let subject = execute(
        api,
        worker,
        &subject_job,
        "completed",
        root,
        "caption-subject",
    )
    .await?;
    let before_cancel = api.json("GET", &dataset_route, Value::Null).await?;
    require(
        subject["result"]["mode"] == "subjectOnly"
            && subject["result"]["variant"] == tier
            && grounded(field(&before_cancel["items"][0]["caption"], "text")?, 0),
        "subject-only result lost its mode, tier or subject",
    )?;
    let canceled_job = api
        .json("POST", &caption_route, body("subjectOnly"))
        .await?;
    let cancel_id = field(&canceled_job, "id")?;
    *ledger.cancel_after_post.lock().unwrap() = Some(CancelOnPost {
        route: format!("/api/v1/jobs/{cancel_id}/progress"),
        message: Some("Preparing training caption job."),
        cancel_url: format!("{}/api/v1/jobs/{cancel_id}/cancel", api.url),
    });
    let start = ledger.sequence.load(Ordering::Relaxed);
    execute(
        api,
        worker,
        &canceled_job,
        "canceled",
        root,
        "caption-canceled",
    )
    .await?;
    let wrote_sidecars = ledger.rows.lock().unwrap().iter().any(|r| {
        r["sequence"].as_u64().is_some_and(|seq| seq >= start)
            && r["method"] == "POST"
            && r["route"]
                .as_str()
                .is_some_and(|p| p.ends_with("/caption-sidecars"))
    });
    let after_cancel = api.json("GET", &dataset_route, Value::Null).await?;
    require(
        !wrote_sidecars
            && before_cancel["items"] == after_cancel["items"]
            && before_cancel["version"] == after_cancel["version"],
        "caption cancellation wrote sidecars or changed the dataset",
    )?;
    let trigger_job = api
        .json("POST", &caption_route, body("triggerOnly"))
        .await?;
    let trigger = execute(
        api,
        worker,
        &trigger_job,
        "completed",
        root,
        "caption-trigger-only",
    )
    .await?;
    let updated = api.json("GET", &dataset_route, Value::Null).await?;
    require(
        trigger["result"]["mode"] == "triggerOnly"
            && trigger["result"].get("modelNameOrPath").is_none()
            && trigger["result"].get("variant").is_none()
            && updated["items"][0]["caption"]["text"] == "loopshape",
        "trigger-only returned a model or wrote anything beyond its trigger words",
    )?;

    for (name, task, workflow, target, style) in [
        (
            "structured-caption",
            "image_caption",
            "image",
            "ideogram_4",
            None,
        ),
        (
            "describe-prose",
            "image_describe",
            "image",
            "z_image",
            Some("prose"),
        ),
        (
            "describe-tags",
            "image_describe",
            "image",
            "illustrious_xl_v1",
            Some("tags"),
        ),
        ("rewrite-image", "rewrite", "image", "z_image", None),
        ("rewrite-video", "rewrite", "video", "wan_2_2", None),
    ] {
        let mut request = json!({"projectId":project,"model":MODEL,"variant":tier,"task":task,"workflow":workflow,"modelId":target});
        let prompt = "A red kite drifts above a field.";
        if task == "rewrite" {
            request.as_object_mut().unwrap().remove("task");
            request["prompt"] = json!(prompt);
        } else {
            request["sourceAssetId"] = json!(asset_id);
        }
        if let Some(style) = style {
            request["captionStyle"] = json!(style);
        }
        let job = api.json("POST", "/api/v1/prompts/refine", request).await?;
        require(
            job["payload"]["modelId"] == target
                && job["payload"]["model"] == MODEL
                && job["payload"]["variant"] == tier,
            "refine confused source or target model",
        )?;
        let run = execute(api, worker, &job, "completed", root, name).await?;
        let result = &run["result"];
        let identity = &result["executionIdentity"];
        let lane = if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        };
        require(
            identity["model"] == MODEL
                && identity["variant"] == tier
                && identity["backend"] == lane
                && identity["provider"] == "native",
            "refine execution identity mismatch",
        )?;
        let text = field(result, "refinedPrompt")?.trim();
        require(
            !text.is_empty() && !text.contains("<think>"),
            "empty output or leaked reasoning",
        )?;
        if task == "rewrite" {
            require(
                result["originalPrompt"] == prompt
                    && text != prompt
                    && text.to_lowercase().contains("kite"),
                "rewrite lost the input subject or returned the input unchanged",
            )?;
        } else {
            require(grounded(text, 0), "caption missed the fixture color")?;
            if task == "image_caption" {
                receipt["structuredCaption"] = json!(text);
            }
            if style == Some("tags") {
                require(
                    text.split(',').filter(|s| !s.trim().is_empty()).count() >= 3,
                    "expected a comma-separated tag list",
                )?;
            }
            if style == Some("prose") {
                require(
                    text.split_whitespace().count() >= 5 && text.contains('.'),
                    "expected sentence-style description",
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn batch_evidence_rejects_wrong_image_partial_and_missing_fields_without_a_model() {
    let dataset = json!({"items":[{"id":"circle"},{"id":"square"}]});
    let row = |id: &str, description: &str| {
        json!({
            "itemId":id, "width":384, "height":256, "failed":false, "notice":null, "error":null,
            "description":description,"pose":"","style":"geometric","camera":"","composition":"centered",
            "lighting":"even","environment":"plain field",
        })
    };
    let good = json!({"total":2, "nextOffset":null, "items":[row("circle","red disk"), row("square","blue square")]});
    check_rows(&good, &dataset, 2).unwrap();
    let mut bad = good.clone();
    bad["items"][0].as_object_mut().unwrap().remove("pose");
    assert!(check_rows(&bad, &dataset, 2).is_err());
    bad = good.clone();
    bad["items"][0]["description"] = json!("a blue square");
    assert!(check_rows(&bad, &dataset, 2).is_err());
    bad = good;
    bad["items"][1]["failed"] = json!(true);
    assert!(check_rows(&bad, &dataset, 2).is_err());
    assert!(check_rows(&bad, &dataset, 1).is_err());
}

#[test]
fn official_bf16_metadata_rejects_wrong_revision_or_file_set_without_cache_bytes() {
    let publisher: Value = serde_json::from_str(PUBLISHER).unwrap();
    let revision = "3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6";
    let mut files: Vec<_> = publisher["files"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let rows = official_files(MODEL, revision, &files).unwrap();
    assert_eq!(rows.len(), 20);
    let shard = rows
        .iter()
        .find(|f| f.relative_path == Path::new("model-00001.safetensors"))
        .unwrap();
    assert_eq!(shard.size_bytes, 2034237592);
    assert_eq!(
        shard.sha256,
        "sha256:8e7d17dd41efe79746bcff1e51a892ec6bf8092887cba88827cefc77b19b3fd3"
    );
    assert!(official_files(MODEL, "main", &files).is_err());
    assert!(official_files("example/foreign", revision, &files).is_err());
    files[0] = files[1].clone();
    assert!(official_files(MODEL, revision, &files).is_err());
    files[0] = "foreign.txt".into();
    assert!(official_files(MODEL, revision, &files).is_err());
}

/// Exercises the same JS review/save/export client and actual Rust routes without any worker.
#[tokio::test]
#[ignore = "explicit metadata-only test using the local Node runtime"]
async fn review_round_trip_without_a_model() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let dataset = json!({"id":"fixture-dataset","name":"Review test","version":2});
    let items = json!([{"itemId":"circle","description":"red circle","width":384,"height":256},{"itemId":"square","description":"blue square","width":384,"height":256}]);
    let outcome = save_review(&url, temp.path(), &items, &dataset, "bf16").await;
    let _ = stop.send(());
    server.await.unwrap().unwrap();
    outcome.unwrap();
}
