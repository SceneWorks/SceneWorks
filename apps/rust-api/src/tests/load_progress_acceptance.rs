//! sc-27984: explicit real-weight acceptance, never part of ordinary CI execution.
//! Uses the shipped API, real worker registration/claim/dispatcher, and frozen manifests.
use super::support::*;
use futures_util::FutureExt;
use sceneworks_worker::test_support as native;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::UNIX_EPOCH,
};

type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing {key}: {v}"))
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_millis()
}
fn save(root: &Path, name: &str, value: &Value) -> Result<()> {
    std::fs::write(
        root.join(name),
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

#[derive(Clone, Default)]
struct HttpLedger {
    sequence: Arc<AtomicU64>,
    rows: Arc<Mutex<Vec<Value>>>,
}
async fn record_http(
    ledger: HttpLedger,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    // Arrival order and receive time precede dispatch, so slow POSTs cannot disguise stale labels.
    let sequence = ledger.sequence.fetch_add(1, Ordering::Relaxed);
    let received = now();
    let (parts, body) = request.into_parts();
    let method = parts.method.to_string();
    let route = parts.uri.path().to_owned();
    let bytes = to_bytes(body, 16 * 1024 * 1024)
        .await
        .expect("bounded owned request");
    let request_value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let response = next
        .run(axum::extract::Request::from_parts(
            parts,
            Body::from(bytes.clone()),
        ))
        .await;
    let (parts, body) = response.into_parts();
    let output = to_bytes(body, 32 * 1024 * 1024)
        .await
        .expect("bounded owned response");
    let response_value: Value = serde_json::from_slice(&output).unwrap_or(Value::Null);
    ledger.rows.lock().expect("HTTP ledger").push(json!({"sequence":sequence,"receivedUnixMs":received,"completedUnixMs":now(),"method":method,"route":route,"request":request_value,"requestBodySha256":sha(&bytes),"response":response_value,"responseBodySha256":sha(&output),"httpStatus":parts.status.as_u16()}));
    axum::response::Response::from_parts(parts, Body::from(output))
}
struct Api {
    client: reqwest::Client,
    url: String,
}
impl Api {
    async fn json(&self, method: &str, route: &str, body: Value) -> Result<Value> {
        let response = self
            .client
            .request(
                method
                    .parse::<reqwest::Method>()
                    .map_err(|e| e.to_string())?,
                format!("{}{route}", self.url),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let value: Value = response.json().await.map_err(|e| e.to_string())?;
        require(
            status.is_success(),
            format!("{method} {route}: {status} {value}"),
        )?;
        Ok(value)
    }
    async fn upload(&self, route: &str, name: &str, mime: &str, bytes: &[u8]) -> Result<Value> {
        let mut body = format!("--LOAD_PROGRESS\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n").into_bytes();
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n--LOAD_PROGRESS--\r\n");
        let response = self
            .client
            .post(format!("{}{route}", self.url))
            .header(
                "content-type",
                "multipart/form-data; boundary=LOAD_PROGRESS",
            )
            .body(body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let value: Value = response.json().await.map_err(|e| e.to_string())?;
        require(status.is_success(), format!("upload {status}: {value}"))?;
        Ok(value)
    }
}
fn model_entry(id: &str) -> Result<Value> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .ok_or("shipped manifest missing")?
        .1;
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).map_err(|e| e.to_string())?;
    manifest["models"]
        .as_array()
        .and_then(|v| v.iter().find(|m| m["id"] == id))
        .cloned()
        .ok_or_else(|| format!("missing shipped {id}"))
}
fn snapshot_inventory(settings: &Settings, model: &str) -> Result<Value> {
    let entry = model_entry(model)?;
    let mut files = Vec::new();
    for source in entry["downloads"]
        .as_array()
        .ok_or("download manifest missing")?
    {
        let repo = field(source, "repo")?;
        let revision = field(source, "revision")?;
        require(
            revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
            "immutable model revision required",
        )?;
        let cache =
            huggingface_repo_cache_path(&settings.data_dir, repo).ok_or("HF cache unavailable")?;
        let snapshot = cache.join("snapshots").join(revision);
        for name in source["files"].as_array().ok_or("declared files missing")? {
            let name = name.as_str().ok_or("invalid manifest filename")?;
            require(
                !name.contains('*') && !name.contains("..") && !Path::new(name).is_absolute(),
                "acceptance requires exact shipped relative filenames",
            )?;
            let path = std::fs::canonicalize(snapshot.join(name)).map_err(|e| {
                format!("provision frozen {repo}@{revision}/{name} before acceptance: {e}")
            })?;
            let metadata = std::fs::metadata(&path).map_err(|e| e.to_string())?;
            require(
                metadata.is_file() && metadata.len() > 0,
                format!("missing {name}"),
            )?;
            files.push(json!({"repo":repo,"revision":revision,"file":name,"resolvedPath":path,"bytes":metadata.len(),"component":source["componentId"]}));
        }
    }
    Ok(json!({"model":model,"manifest":entry,"files":files}))
}

fn progress_messages(rows: &[Value], job: &str) -> Vec<String> {
    let route = format!("/api/v1/jobs/{job}/progress");
    let mut rows: Vec<_> = rows
        .iter()
        .filter(|r| {
            r["method"] == "POST"
                && r["route"] == route
                && r["httpStatus"].as_u64().is_some_and(|s| s < 300)
        })
        .collect();
    rows.sort_by_key(|r| r["sequence"].as_u64());
    rows.into_iter()
        .filter_map(|r| r["request"]["message"].as_str().map(str::to_owned))
        .collect()
}
fn has_bytes(message: &str) -> bool {
    message.contains(" — ")
        && message.contains(" / ")
        && [" b", " kb", " mb", " gb", " tb"]
            .iter()
            .any(|u| message.contains(u))
}
fn validate_progress(messages: &[String], case: &str) -> Result<()> {
    let image = case.starts_with("image_");
    let training = case == "training";
    let lowered: Vec<_> = messages.iter().map(|m| m.to_lowercase()).collect();
    let load = lowered
        .iter()
        .position(|m| m.contains("loading ") && has_bytes(m))
        .ok_or("no actual byte-bearing loading POST")?;
    if image {
        let checking = lowered
            .iter()
            .position(|m| {
                (m.contains("checking model files (") || m.contains("verifying "))
                    && m.chars().any(|c| c.is_ascii_digit())
            })
            .ok_or("missing checking/verifying counts or bytes")?;
        let component = |name: &str| {
            lowered
                .iter()
                .position(|m| m.contains(&format!("loading {name}")) && has_bytes(m))
                .ok_or_else(|| format!("missing {name} byte progress"))
        };
        let text = component("text encoder")?;
        let vae = component("vae")?;
        let transformer = component("transformer")?;
        let prompt = lowered
            .iter()
            .position(|m| m.contains("encoding prompt"))
            .ok_or("missing visible universal prompt encoding")?;
        let step = lowered
            .iter()
            .position(|m| m.contains("step "))
            .ok_or("missing real denoise steps")?;
        // MLX materializes encoder+VAE in load; Candle defers VAE weights until decode.
        // Both encode conditioning before constructing the first lazy transformer renderer.
        let ordered = if case == "image_cuda" {
            checking < text
                && text < prompt
                && prompt < transformer
                && transformer < step
                && step < vae
        } else {
            checking < text
                && text < vae
                && vae < prompt
                && prompt < transformer
                && transformer < step
        };
        require(
            ordered,
            format!("required {case} MOSS sequence differs; inspect recorded POSTs: {messages:?}"),
        )?;
    }
    let compute = lowered
        .iter()
        .enumerate()
        .find_map(|(index, m)| {
            let is_compute = if training {
                m.contains("training step 1 of 1")
            } else {
                m.contains("encoding prompt")
                    || m.contains("step ")
                    || m.contains("synthes")
                    || m.contains("decoding")
                    || m.contains("generating audio (")
            };
            (index > load && is_compute).then_some(index)
        })
        .ok_or("missing compute progress after loading")?;
    require(load < compute, "compute precedes byte-bearing load")?;
    // Invalidate IO labels at EVERY compute transition, including denoising after a lazy
    // transformer read. Distinct later lazy reads (Candle VAE decode) remain truthful.
    let mut seen_io = std::collections::BTreeSet::<&str>::new();
    let mut obsolete_io = std::collections::BTreeSet::<&str>::new();
    for message in &lowered {
        if message.contains("loading ")
            || message.contains("verifying ")
            || message.contains("checking model files")
        {
            require(
                !obsolete_io.contains(message.as_str()),
                "earlier IO label reappeared after compute",
            )?;
            seen_io.insert(message.as_str());
        }
        if message.contains("encoding prompt")
            || message.contains("step ")
            || message.contains("generating audio (")
            || message.contains("decoding")
            || message.contains("synthes")
        {
            obsolete_io.extend(seen_io.iter().copied());
        }
    }
    Ok(())
}

fn reference_wav() -> Vec<u8> {
    // Two seconds of deterministic voiced harmonics, public synthetic input; model execution is real.
    let samples: Vec<i16> = (0..48_000)
        .map(|i| {
            let t = i as f64 / 24_000.0;
            ((std::f64::consts::TAU * 170.0 * t).sin() * 6_000.0
                + (std::f64::consts::TAU * 340.0 * t).sin() * 2_000.0) as i16
        })
        .collect();
    let size = (samples.len() * 2) as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + size).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&24_000_u32.to_le_bytes());
    bytes.extend_from_slice(&48_000_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&size.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}
async fn submit(api: &Api, root: &Path, project: &str, case: &str) -> Result<Value> {
    if case.starts_with("image_") {
        api.json("POST", "/api/v1/image/jobs", json!({"projectId":project,"model":"moss_image_sft","prompt":"A red ceramic cup on a white table","width":256,"height":256,"count":1,"seed":26325,"advanced":{"steps":2,"guidanceScale":1.0,"mossImage":{"acceleration":"off","max_sequence_length":64}}})).await
    } else if case == "audio" {
        let wav = reference_wav();
        std::fs::write(root.join("reference.wav"), &wav).map_err(|e| e.to_string())?;
        let asset = api
            .upload(
                &format!("/api/v1/projects/{project}/assets"),
                "reference.wav",
                "audio/wav",
                &wav,
            )
            .await?;
        api.json("POST", "/api/v1/audio/jobs", json!({"projectId":project,"model":"chatterbox_tts","prompt":"The red cup is on the table.","referenceAudioAssetId":field(&asset,"id")?,"seed":27980})).await
    } else {
        let image = image::RgbImage::from_fn(256, 256, |x, y| {
            if x.abs_diff(128).pow(2) + y.abs_diff(128).pow(2) < 64_u32.pow(2) {
                image::Rgb([210, 30, 25])
            } else {
                image::Rgb([235, 235, 220])
            }
        });
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        std::fs::write(root.join("training-source.png"), png.get_ref())
            .map_err(|e| e.to_string())?;
        let upload = api
            .upload(
                &format!("/api/v1/projects/{project}/training/uploads"),
                "source.png",
                "image/png",
                png.get_ref(),
            )
            .await?;
        let dataset = api.json("POST", &format!("/api/v1/projects/{project}/training/datasets"), json!({"name":"load progress single sample","modality":"image","items":[{"path":upload["file"]["path"],"displayName":"source.png","caption":{"text":"sksmoss red circle"},"width":256,"height":256}]})).await?;
        let targets = api
            .json("GET", "/api/v1/training/targets", Value::Null)
            .await?;
        let target = targets["targets"]
            .as_array()
            .and_then(|v| v.iter().find(|t| t["id"] == "moss_image_sft_lora"))
            .ok_or("shipped SFT LoRA target missing")?;
        let mut config = target["defaults"].clone();
        for (key, value) in [
            ("steps", json!(1)),
            ("batchSize", json!(1)),
            ("gradientAccumulation", json!(1)),
            ("saveEvery", json!(1)),
            ("resolution", json!(256)),
            ("rank", json!(4)),
            ("alpha", json!(6.0)),
            ("learningRate", json!(0.0001)),
            ("seed", json!(1789)),
            ("triggerWord", json!("sksmoss")),
        ] {
            config[key] = value;
        }
        if !config["advanced"].is_object() {
            config["advanced"] = json!({});
        }
        for (key, value) in [
            ("networkType", json!("lora")),
            ("mixedPrecision", json!("bf16")),
            ("cacheLatents", json!(true)),
            ("cacheTextEmbeddings", json!(true)),
            ("sampleEvery", json!(0)),
            ("gradientCheckpointing", json!(false)),
            ("lrWarmupSteps", json!(0)),
        ] {
            config["advanced"][key] = value;
        }
        api.json("POST", &format!("/api/v1/projects/{project}/training/jobs"), json!({"targetId":"moss_image_sft_lora","datasetId":dataset["id"],"config":config,"outputName":"load-progress-one-update","dryRun":false})).await
    }
}
async fn output_evidence(
    api: &Api,
    root: &Path,
    project: &str,
    completed: &Value,
    training: bool,
) -> Result<Value> {
    if training {
        require(
            completed["result"]["stepsCompleted"] == 1
                && completed["result"]["trainedArtifactExported"] == true,
            "one-update trainer did not export",
        )?;
        let output = PathBuf::from(field(&completed["result"], "outputPath")?);
        let bytes = std::fs::read(&output).map_err(|e| e.to_string())?;
        require(bytes.len() > 8, "empty adapter export")?;
        // Keep trained weights in owned state; upload only their identity with the acceptance logs.
        Ok(
            json!({"path":output,"sha256":sha(&bytes),"bytes":bytes.len(),"loraId":completed["result"]["loraId"]}),
        )
    } else {
        let assets = completed["result"]["assets"]
            .as_array()
            .ok_or("completed job assets absent")?;
        require(assets.len() == 1, "expected one generated asset")?;
        let asset = &assets[0];
        let relative = field(asset, "relativePath")?;
        let response = api
            .client
            .get(format!(
                "{}/api/v1/projects/{project}/files/{relative}",
                api.url
            ))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        require(
            response.status().is_success(),
            "generated asset not served by product",
        )?;
        let bytes = response.bytes().await.map_err(|e| e.to_string())?;
        let audio = relative.ends_with(".wav");
        let mut rms = None;
        if audio {
            require(
                bytes.starts_with(b"RIFF") && bytes.len() > 44 && &bytes[8..12] == b"WAVE",
                "invalid generated WAV",
            )?;
            let data = bytes
                .windows(4)
                .position(|chunk| chunk == b"data")
                .ok_or("WAV data chunk absent")?;
            let size = bytes
                .get(data + 4..data + 8)
                .ok_or("truncated WAV data length")?;
            let length =
                u32::from_le_bytes(size.try_into().map_err(|_| "invalid WAV length")?) as usize;
            let samples = bytes
                .get(data + 8..data + 8 + length)
                .ok_or("truncated WAV samples")?;
            require(
                samples.len() >= 2 && samples.len() % 2 == 0,
                "invalid PCM16 output",
            )?;
            let sum = samples
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (f64::from(i16::from_le_bytes([p[0], p[1]])) / 32768.0).powi(2))
                .sum::<f64>();
            let measured = (sum / (samples.len() / 2) as f64).sqrt();
            require(
                measured.is_finite() && measured > 1e-4,
                "generated audio is silent",
            )?;
            rms = Some(measured);
        } else {
            let decoded = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
            require(
                decoded.width() == 256 && decoded.height() == 256,
                "wrong generated dimensions",
            )?;
        }
        if !audio {
            let report: native::gen_core::moss_image::MossImageExecutionReport =
                serde_json::from_value(
                    asset["recipe"]["rawAdapterSettings"]["mossImageExecutionReport"].clone(),
                )
                .map_err(|e| e.to_string())?;
            let backend =
                serde_json::to_value(report.data().execution_device).map_err(|e| e.to_string())?;
            require(
                backend
                    == if cfg!(target_os = "macos") {
                        "mlx"
                    } else {
                        "cuda"
                    },
                "image receipt does not prove selected backend",
            )?;
        }
        let name = if audio { "output.wav" } else { "output.png" };
        std::fs::write(root.join(name), &bytes).map_err(|e| e.to_string())?;
        Ok(json!({"file":name,"sha256":sha(&bytes),"bytes":bytes.len(),"rms":rms,"asset":asset}))
    }
}
async fn run(case: &str) {
    assert_eq!(
        std::env::var("LOAD_PROGRESS_CASE").as_deref(),
        Ok(case),
        "explicit case admission required"
    );
    assert_eq!(std::env::var("HF_HUB_OFFLINE").as_deref(), Ok("1"));
    assert_eq!(std::env::var("TRANSFORMERS_OFFLINE").as_deref(), Ok("1"));
    #[cfg(target_os = "macos")]
    assert_eq!(
        std::env::var("LOAD_PROGRESS_MAC_GO").as_deref(),
        Ok("true"),
        "Michael's explicit GPU go required"
    );
    let provenance: Value = serde_json::from_str(include_str!(
        "../../../../config/inference-third-party-source.json"
    ))
    .unwrap();
    assert_eq!(
        std::env::var("LOAD_PROGRESS_INFERENCE_SHA").ok().as_deref(),
        provenance["inferenceRevision"].as_str(),
        "exact compiled inference provenance required"
    );
    let root = PathBuf::from(
        std::env::var("LOAD_PROGRESS_OUTPUT").expect("owned outside-checkout output required"),
    );
    assert!(root.is_absolute() && !root.starts_with(env!("CARGO_MANIFEST_DIR")));
    std::fs::create_dir_all(&root).unwrap();
    assert!(!root.join("receipt.json").exists(), "fresh output required");
    let ledger = HttpLedger::default();
    let mut receipt = json!({"schemaVersion":1,"case":case,"status":"running","startedUnixMs":now(),"sourceSha":std::env::var("LOAD_PROGRESS_SOURCE_SHA").ok(),"inferenceSha":std::env::var("LOAD_PROGRESS_INFERENCE_SHA").ok()});
    save(&root, "receipt.json", &receipt).unwrap();
    let state_root = PathBuf::from(
        std::env::var("LOAD_PROGRESS_STATE_ROOT").expect("outside-evidence state root required"),
    );
    assert!(state_root.is_absolute() && !state_root.starts_with(&root));
    std::fs::create_dir_all(&state_root).unwrap();
    let temp = tempfile::tempdir_in(&state_root).unwrap();
    let mut settings = test_settings(&temp);
    let state = temp.keep();
    settings.host_os = std::env::consts::OS.into();
    super::moss_training_acceptance::seed_acceptance_manifests(&settings).unwrap();
    let app = create_app(settings.clone()).unwrap();
    let tap = ledger.clone();
    let app = app.layer(axum::middleware::from_fn(move |request, next| {
        record_http(tap.clone(), request, next)
    }));
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
    let api = Api {
        client: reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap(),
        url: url.clone(),
    };
    let outcome = std::panic::AssertUnwindSafe(async {
        let model = if case == "audio" {
            "chatterbox_tts"
        } else {
            "moss_image_sft"
        };
        receipt["weights"] = snapshot_inventory(&settings, model)?;
        receipt["statePath"] = json!(state);
        let gpu = if cfg!(target_os = "macos") {
            "mlx".to_owned()
        } else {
            std::env::var("SCENEWORKS_GPU_ID").map_err(|_| "explicit physical GPU_ID required")?
        };
        let mut worker = super::moss_training_acceptance::worker_settings(url, &settings, gpu);
        let registered = native::register_real_acceptance_worker(&worker)
            .await
            .map_err(|e| e.to_string())?;
        worker.gpu_id = registered.gpu_id.clone();
        receipt["worker"] = serde_json::to_value(registered).map_err(|e| e.to_string())?;
        let project = api
            .json(
                "POST",
                "/api/v1/projects",
                json!({"name":"sc27980 terminal load progress"}),
            )
            .await?;
        let project = field(&project, "id")?;
        let submitted = submit(&api, &root, project, case).await?;
        receipt["submittedJob"] = submitted.clone();
        save(&root, "receipt.json", &receipt)?;
        let id = field(&submitted, "id")?;
        let claimed = api
            .json(
                "POST",
                "/api/v1/jobs/claim",
                json!({"workerId":worker.worker_id}),
            )
            .await?;
        require(claimed["job"]["id"] == id, "claim selected another job")?;
        let job = serde_json::from_value(claimed["job"].clone()).map_err(|e| e.to_string())?;
        // Await normal dispatcher teardown even on failure. No timeout drops a live native task.
        native::run_acceptance_utility_job(&worker, job).await;
        let completed = api
            .json("GET", &format!("/api/v1/jobs/{id}"), Value::Null)
            .await?;
        receipt["completedJob"] = completed.clone();
        save(&root, "receipt.json", &receipt)?;
        require(
            completed["status"] == "completed",
            format!("real worker failed: {completed}"),
        )?;
        receipt["output"] =
            output_evidence(&api, &root, project, &completed, case == "training").await?;
        let rows = ledger.rows.lock().unwrap().clone();
        let messages = progress_messages(&rows, id);
        receipt["messages"] = json!(messages);
        validate_progress(&messages, case)?;
        Ok::<_, String>(())
    })
    .catch_unwind()
    .await;
    let _ = stop.send(());
    let joined = server.await;
    let failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e),
        Err(panic) => Some(
            panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "acceptance panic".into()),
        ),
    };
    let mut rows = ledger.rows.lock().unwrap().clone();
    rows.sort_by_key(|r| r["sequence"].as_u64());
    save(&root, "worker-http.json", &json!(rows)).unwrap();
    receipt["status"] = json!(if failure.is_none() && matches!(joined, Ok(Ok(()))) {
        "passed"
    } else {
        "failed"
    });
    receipt["failure"] = json!(failure);
    receipt["serverJoined"] = json!(matches!(joined, Ok(Ok(()))));
    receipt["finishedUnixMs"] = json!(now());
    save(&root, "receipt.json", &receipt).unwrap();
    assert_eq!(receipt["status"], "passed", "{}", receipt["failure"]);
}
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "sc27984 explicit manual real-weight MLX acceptance only"]
async fn image_mlx() {
    run("image_mlx").await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "sc27984 explicit manual real-weight CUDA acceptance only"]
async fn image_cuda() {
    run("image_cuda").await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "sc27984 explicit manual real-weight outside-cache audio acceptance only"]
async fn audio() {
    run("audio").await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "sc27984 explicit manual real-weight one-update trainer acceptance only"]
async fn training() {
    run("training").await;
}

#[test]
fn progress_sequence_requires_bytes_and_current_phase() {
    let mlx: Vec<String> = [
        "Checking model files (1/1).",
        "Loading text encoder — 1.0 / 2.0 GB",
        "Loading vae — 1.0 / 2.0 GB",
        "Image 1/1 — Encoding prompt.",
        "Image 1/1 — Encoding prompt.",
        "Loading transformer — 1.0 / 2.0 GB",
        "Image 1/1 — step 1 of 2",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    validate_progress(&mlx, "image_mlx").unwrap();
    let cuda = vec![
        mlx[0].clone(),
        mlx[1].clone(),
        mlx[3].clone(),
        mlx[4].clone(),
        mlx[5].clone(),
        mlx[6].clone(),
        mlx[2].clone(),
    ];
    validate_progress(&cuda, "image_cuda").unwrap();
    assert!(validate_progress(&mlx, "image_cuda").is_err());
    assert!(validate_progress(&cuda, "image_mlx").is_err());
    // A stale transformer label after denoising must fail, even though it first appeared
    // after prompt encoding. Likewise the completed encoder label cannot return.
    for obsolete in [mlx[1].clone(), mlx[5].clone()] {
        let mut stale = mlx.clone();
        stale.push(obsolete);
        assert!(validate_progress(&stale, "image_mlx").is_err());
    }
    let mut wrong_order = mlx;
    wrong_order.swap(3, 5);
    assert!(validate_progress(&wrong_order, "image_mlx").is_err());
    assert!(validate_progress(
        &["Loading weights".into(), "Training step 1 of 1".into()],
        "training"
    )
    .is_err());
}
