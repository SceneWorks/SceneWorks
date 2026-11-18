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
    let tier = field(&job["payload"], "variant")?;
    native::run_bound_acceptance_job(worker, id, |route| {
        check_binding(route, tier)?;
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
            row.to_string().to_lowercase().contains(SCENES[i].1),
            "description lost the fixture's dominant color",
        )?;
    }
    Ok(())
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
    *ledger.cancel_after_item.lock().unwrap() = Some((
        format!("/api/v1/jobs/{id}/result-items"),
        format!("{}/api/v1/jobs/{id}/cancel", api.url),
    ));
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
    for (i, (_, color, _)) in SCENES.iter().enumerate() {
        let item = &updated["items"][i];
        require(
            item["id"] == dataset["items"][i]["id"]
                && item["caption"]["source"] == "auto"
                && field(&item["caption"], "text")?.starts_with("loopshape")
                && field(&item["caption"], "text")?
                    .to_lowercase()
                    .contains(*color),
            "missing grounded auto-caption on its source image",
        )?;
    }

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
            require(
                text.to_lowercase().contains("red"),
                "caption missed the fixture color",
            )?;
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
