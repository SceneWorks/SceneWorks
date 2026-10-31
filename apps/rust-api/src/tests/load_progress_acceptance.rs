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
// The frozen Chatterbox path loads reference/tokenizer weights, encodes the prompt,
// samples AR tokens, constructs the renderer, then runs its three stages and HiFT.
// Each renderer submodule opens the same S3Gen file independently. Rounded HTTP
// labels carry no operation ID: equality is not evidence of an obsolete operation.
fn validate_audio_epochs(messages: &[String]) -> Result<()> {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Phase {
        Preparation,
        Prompt,
        Autoregressive,
        Renderer,
        Stages,
        Decoding,
        PostProcessing,
        Saving,
        Output,
    }
    let mut phase = Phase::Preparation;
    let mut last_step = None;
    for message in messages {
        let io = message.starts_with("loading ") || message.starts_with("verifying ");
        if io {
            require(
                phase == Phase::Preparation
                    || (!message.contains("ve.safetensors")
                        && !message.contains("t3 cfg.safetensors")),
                "completed audio conditioning IO returned after prompt encoding",
            )?;
            if message.contains("s3gen.safetensors") {
                require(
                    phase <= Phase::Renderer,
                    "renderer IO returned after S3Gen execution began",
                )?;
                // Reference tokenization also reads S3Gen before the prompt. After an
                // observed prompt/AR phase the same file belongs to renderer construction,
                // even if every intervening AR callback coalesced. With neither observed,
                // these operation identities cannot be recovered from rounded HTTP labels.
                if phase >= Phase::Prompt {
                    phase = Phase::Renderer;
                }
            }
            if message.contains("perth implicit.safetensors") {
                require(
                    phase <= Phase::PostProcessing,
                    "postprocessing IO returned after audio saving began",
                )?;
                phase = Phase::PostProcessing;
            }
        }
        if message.contains("encoding prompt") {
            require(
                phase <= Phase::Prompt,
                "audio prompt label returned after sampling began",
            )?;
            phase = Phase::Prompt;
        }
        require(
            phase == Phase::Preparation
                || (!message.contains("synthes") && message != "preparing audio."),
            "audio preparation label returned after sampling began",
        )?;
        if let Some(step) = message
            .strip_prefix("generating step ")
            .and_then(|s| s.strip_suffix('.'))
        {
            let (current, total) = step.split_once('/').ok_or("audio step lacks counters")?;
            let current: u32 = current.parse().map_err(|_| "invalid audio step")?;
            let total: u32 = total.parse().map_err(|_| "invalid audio total")?;
            require(
                phase <= Phase::Stages,
                "audio step returned after output finishing began",
            )?;
            require(
                total > 0 && current > 0 && current <= total,
                "invalid audio step bounds",
            )?;
            let next = if total == 3 {
                Phase::Stages
            } else {
                Phase::Autoregressive
            };
            require(
                phase <= next,
                "autoregressive step returned during renderer execution",
            )?;
            if let Some((previous, previous_total)) = last_step {
                require(
                    previous_total != total || current >= previous,
                    "audio step counter regressed",
                )?;
            }
            phase = next;
            last_step = Some((current, total));
        }
        // A later observed phase retires earlier states even when intermediate
        // callbacks coalesce. Duplicate snapshots of the current phase remain valid.
        if message.contains("decoding") {
            require(
                phase <= Phase::Decoding,
                "audio decoding returned after postprocessing began",
            )?;
            phase = Phase::Decoding;
        }
        if message == "finishing output." {
            require(
                phase <= Phase::PostProcessing,
                "postprocessing returned after audio saving began",
            )?;
            phase = Phase::PostProcessing;
        }
        if message == "saving audio." {
            require(
                phase <= Phase::Saving,
                "audio saving returned after output completion",
            )?;
            phase = Phase::Saving;
        }
        if message == "generated audio." {
            phase = Phase::Output;
        }
    }
    Ok(())
}

fn validate_progress(messages: &[String], case: &str) -> Result<()> {
    let image = case.starts_with("image_");
    let training = case == "training";
    let audio = case == "audio";
    let lowered: Vec<_> = messages.iter().map(|m| m.to_lowercase()).collect();
    if audio {
        validate_audio_epochs(&lowered)?;
    }
    let load = lowered
        .iter()
        .position(|m| m.contains("loading ") && has_bytes(m))
        .ok_or("no actual byte-bearing loading POST")?;
    let mut renderer_start = None;
    let mut vae_decode_start = None;
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
        let prompt = lowered
            .iter()
            .position(|m| m.contains("encoding prompt"))
            .ok_or("missing visible universal prompt encoding")?;
        let step = lowered
            .iter()
            .position(|m| m.contains("step "))
            .ok_or("missing real denoise steps")?;
        // Candle first authenticates source dtypes/headers, streaming each file before
        // any prompt work. Its actual renderer reload is the transformer read AFTER
        // prompt encoding; lazy VAE materialization is the read AFTER denoising.
        let in_window = |name: &str, after: usize, before: usize| {
            lowered
                .iter()
                .enumerate()
                .find_map(|(index, message)| {
                    (index > after
                        && index < before
                        && message.starts_with(&format!("loading {name}"))
                        && has_bytes(message))
                    .then_some(index)
                })
                .ok_or_else(|| format!("missing {name} byte progress in its compute window"))
        };
        let transformer = in_window("transformer", prompt, step)?;
        renderer_start = Some(transformer);
        let ordered = if case == "image_cuda" {
            vae_decode_start = Some(in_window("vae", step, lowered.len())?);
            checking < text && text < prompt
        } else {
            checking < text && text < vae && vae < prompt
        };
        require(
            ordered,
            format!("required {case} MOSS sequence differs; inspect recorded POSTs: {messages:?}"),
        )?;
        require(
            !lowered[transformer + 1..]
                .iter()
                .any(|message| message.contains("encoding prompt")),
            "prompt encoding label returned after transformer loading began",
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
    // Compute invalidates prior IO. A subsequent byte-bearing load establishes a new
    // component epoch: read_hashed_file loads bytes, then hashes those bytes afresh, so
    // its verification labels may equal labels from the earlier cold source seal.
    let mut seen_io = std::collections::BTreeSet::<&str>::new();
    let mut obsolete_io = std::collections::BTreeSet::<&str>::new();
    let mut computed = false;
    let mut fresh_component = None;
    let mut read_progress = None;
    let mut denoising_started = false;
    for (index, message) in lowered.iter().enumerate() {
        let renderer_started = renderer_start.is_some_and(|start| index >= start);
        if vae_decode_start.is_some_and(|start| index > start) {
            require(
                !message.contains("step "),
                "denoising label returned after VAE decode loading began",
            )?;
        }
        let loading = message.strip_prefix("loading ");
        let verifying = message.strip_prefix("verifying ");
        let checking = message.contains("checking model files");
        if image && renderer_started {
            require(
                !checking
                    && (!denoising_started
                        || (!message.starts_with("loading transformer")
                            && !message.starts_with("verifying transformer")))
                    && !message.starts_with("loading text encoder")
                    && !message.starts_with("verifying text encoder")
                    && ((case == "image_cuda" && denoising_started)
                        || (!message.starts_with("loading vae")
                            && !message.starts_with("verifying vae"))),
                "completed component IO returned during transformer execution",
            )?;
        }
        if loading.is_some() || verifying.is_some() || checking {
            if let Some(label) = loading.filter(|_| !image && !audio && has_bytes(message)) {
                fresh_component = Some(
                    label
                        .split(" (")
                        .next()
                        .unwrap()
                        .split(" — ")
                        .next()
                        .unwrap(),
                );
                read_progress = None;
            }
            if let Some(label) = verifying.or(loading.filter(|_| image || audio)) {
                let (source, progress) = label.split_once(" — ").ok_or("IO report lacks bytes")?;
                let component = source.split(" (").next().unwrap();
                if verifying.is_some() {
                    require(
                        !computed || fresh_component == Some(component),
                        "verification returned after compute without a fresh component load",
                    )?;
                } else if has_bytes(message) {
                    fresh_component = Some(component);
                }
                let current = progress.split(" / ").next().unwrap();
                let unit = progress.rsplit_once(' ').ok_or("IO report lacks units")?.1;
                let amount: f64 = current
                    .split_whitespace()
                    .next()
                    .ok_or("IO report lacks byte count")?
                    .parse()
                    .map_err(|_| "invalid IO byte count")?;
                let scale = match unit {
                    "b" | "bytes" => 1.,
                    "kb" => 1_000.,
                    "mb" => 1_000_000.,
                    "gb" => 1_000_000_000.,
                    "tb" => 1_000_000_000_000.,
                    _ => return Err("unknown IO byte unit".into()),
                };
                let bytes = amount * scale;
                let identity = (verifying.is_some(), source);
                if let Some((previous_source, previous_bytes)) = read_progress {
                    require(
                        previous_source != identity
                            || bytes >= previous_bytes
                            // Independent S3Gen constructors read the same file, and
                            // coalescing can omit their initial zero. Their construction
                            // window is bounded above; HTTP cannot identify each reader.
                            || (audio && loading.is_some() && source.contains("s3gen.safetensors")),
                        "IO bytes regressed within a read epoch",
                    )?;
                }
                read_progress = Some((identity, bytes));
            }
            // Image epochs are delimited by actual compute windows above. Their fresh
            // reads may repeat rounded byte strings from cold preparation. Audio epochs
            // are checked above; training retains the existing stale-label check.
            if (!image && !audio && verifying.is_none()) || checking {
                require(
                    !obsolete_io.contains(message.as_str()),
                    "earlier IO label reappeared after compute",
                )?;
            }
            seen_io.insert(message.as_str());
        }
        if message.contains("encoding prompt")
            || message.contains("step ")
            || message.contains("generating audio (")
            || message.contains("decoding")
            || message.contains("synthes")
        {
            obsolete_io.extend(seen_io.iter().copied());
            computed = true;
            denoising_started |= image && message.contains("step ");
            fresh_component = None;
            read_progress = None;
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
fn asset_file_path(asset: &Value) -> Result<&str> {
    field(&asset["file"], "path")
}

fn training_export_evidence(completed: &Value) -> Result<Value> {
    use sceneworks_core::contracts::JobStatus;

    let status: JobStatus =
        serde_json::from_value(completed["status"].clone()).map_err(|e| e.to_string())?;
    require(
        status == JobStatus::Completed && completed["result"]["stepsCompleted"] == 1,
        "one-update trainer did not complete",
    )?;
    let output = PathBuf::from(field(&completed["result"], "outputPath")?);
    let bytes = std::fs::read(&output).map_err(|e| e.to_string())?;
    require(bytes.len() > 8, "empty adapter export")?;
    // Keep trained weights in owned state; upload only their identity with the acceptance logs.
    Ok(
        json!({"path":output,"sha256":sha(&bytes),"bytes":bytes.len(),"loraId":completed["result"]["loraId"]}),
    )
}

async fn output_evidence(
    api: &Api,
    root: &Path,
    project: &str,
    completed: &Value,
    training: bool,
) -> Result<Value> {
    if training {
        training_export_evidence(completed)
    } else {
        let assets = completed["result"]["assets"]
            .as_array()
            .ok_or("completed job assets absent")?;
        require(assets.len() == 1, "expected one generated asset")?;
        let asset = &assets[0];
        let relative = asset_file_path(asset)?;
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
    let mut stale_prompt_cases = Vec::new();
    for (case, valid) in [("image_mlx", &mlx), ("image_cuda", &cuda)] {
        let transformer = valid
            .iter()
            .position(|message| message.contains("Loading transformer"))
            .unwrap();
        // The encoder can still read a new weight range while encoding. Repeated positive /
        // negative encoding reports before renderer construction remain legitimate.
        let mut encoding_read = valid.clone();
        encoding_read.insert(transformer, "Loading text encoder — 1.5 / 2.0 GB".into());
        validate_progress(&encoding_read, case).unwrap();
        // Regression: an IO scope ending during lazy renderer construction restored the old
        // EncodingPrompt label, after all prompt encoding had already completed.
        let mut stale_prompt = valid.clone();
        stale_prompt.insert(transformer + 1, "Image 1/1 — Encoding prompt.".into());
        if validate_progress(&stale_prompt, case).is_ok() {
            stale_prompt_cases.push(case);
        }
    }
    assert!(
        stale_prompt_cases.is_empty(),
        "stale prompt replay accepted for {stale_prompt_cases:?}"
    );
    let mut wrong_order = mlx;
    wrong_order.swap(3, 5);
    assert!(validate_progress(&wrong_order, "image_mlx").is_err());
    assert!(validate_progress(
        &["Loading weights".into(), "Training step 1 of 1".into()],
        "training"
    )
    .is_err());
}

#[test]
fn verification_requires_a_fresh_component_epoch_after_compute() {
    let seal = "Verifying transformer (shard.safetensors) — 0.3 / 5.0 GB";
    let mut mlx: Vec<String> = [
        "Checking model files (0/1).",
        seal,
        "Loading text encoder — 1.0 / 2.0 GB",
        "Loading vae — 1.0 / 2.0 GB",
        "Encoding prompt.",
        "Loading transformer (shard.safetensors) — 1.9 / 5.0 GB",
        seal,
        "Verifying transformer (shard.safetensors) — 0.9 / 5.0 GB",
        "Generating step 1/2.",
        "Decoding.",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    // Same source and rounded byte count are legitimate in a new load -> hash epoch.
    validate_progress(&mlx, "image_mlx").unwrap();
    for (position, message) in [
        (5, seal), // Hash replay before the new transformer load.
        (8, seal), // Bytes regress from 0.9 to 0.3 within the current hash.
        (9, seal), // Hash replay after denoising, without a fresh read.
        (6, "Verifying text encoder — 1.0 / 2.0 GB"),
        (6, "Checking model files (0/1)."),
        (6, "Loading vae — 1.5 / 2.0 GB"),
    ] {
        let mut stale = mlx.clone();
        stale.insert(position, message.into());
        assert!(validate_progress(&stale, "image_mlx").is_err(), "{message}");
    }
    let mut late_transformer = mlx.clone();
    late_transformer.insert(
        9,
        "Loading transformer (unseen-shard.safetensors) — 0.1 / 5.0 GB".into(),
    );
    late_transformer.insert(
        10,
        "Verifying transformer (unseen-shard.safetensors) — 0.1 / 5.0 GB".into(),
    );
    assert!(validate_progress(&late_transformer, "image_mlx").is_err());
    // Candle's lazy VAE decode is another real epoch after denoising, even when
    // its hash count equals the initial seal's count for the same source.
    mlx.remove(3);
    let vae_seal = "Verifying vae — 0.1 / 2.0 GB";
    mlx.insert(1, vae_seal.into());
    mlx.push("Loading vae — 1.0 / 2.0 GB".into());
    mlx.push(vae_seal.into());
    validate_progress(&mlx, "image_cuda").unwrap();
}

#[test]
fn candle_preparation_and_materialization_have_distinct_truthful_read_epochs() {
    let messages: Vec<String> = [
        "Checking model files (0/1).",
        "Verifying transformer (shard.safetensors) — 0.3 / 5.0 GB",
        "Loading transformer (shard.safetensors) — 0.1 / 5.0 GB",
        "Loading transformer (shard.safetensors) — 0.7 / 5.0 GB",
        "Loading text encoder — 1.0 / 2.0 GB",
        "Loading vae — 0.0 / 249.5 MB",
        "Loading text encoder — 1.0 / 2.0 GB",
        "Encoding prompt.",
        "Preparing generation.",
        "Loading transformer (shard.safetensors) — 0.1 / 5.0 GB",
        "Loading transformer (shard.safetensors) — 0.7 / 5.0 GB",
        "Generating step 1/2.",
        "Generating step 2/2.",
        "Loading vae — 25.2 / 249.5 MB",
        "Preparing generation.",
        "Decoding.",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    validate_progress(&messages, "image_cuda").unwrap();
    // Cold source validation may read identical bytes before materialization.
    // Equality across epochs is legitimate; a regression within one read is not.
    for (position, stale) in [
        (10, "Loading text encoder — 1.0 / 2.0 GB"),
        (10, "Encoding prompt."),
        (11, "Loading transformer (shard.safetensors) — 0.1 / 5.0 GB"),
        (
            13,
            "Loading transformer (unseen.safetensors) — 0.1 / 5.0 GB",
        ),
        (14, "Generating step 2/2."),
        (
            14,
            "Verifying transformer (shard.safetensors) — 0.3 / 5.0 GB",
        ),
    ] {
        let mut replay = messages.clone();
        replay.insert(position, stale.into());
        assert!(
            validate_progress(&replay, "image_cuda").is_err(),
            "accepted stale {stale}"
        );
    }
    let mut no_renderer_read = messages.clone();
    no_renderer_read.drain(9..11);
    assert!(validate_progress(&no_renderer_read, "image_cuda").is_err());
    let mut no_decode_read = messages;
    no_decode_read.remove(13);
    assert!(validate_progress(&no_decode_read, "image_cuda").is_err());
}

#[test]
fn audio_reader_epochs_and_output_finishing_are_distinct() {
    // Pure source-shaped expectation, not a substitute for a new native trace.
    let expected: Vec<String> = [
        "Loading s3gen.safetensors — 0.1 / 1.1 GB",
        "Synthesizing audio.",
        "Loading t3 cfg.safetensors — 0.1 / 2.1 GB",
        "Encoding prompt.",
        "Generating step 13/4096.",
        "Generating step 62/4096.",
        "Loading s3gen.safetensors — 0.3 / 1.1 GB",
        "Loading s3gen.safetensors — 0.0 / 1.1 GB",
        "Loading s3gen.safetensors — 0.1 / 1.1 GB",
        "Preparing generation.",
        "Loading s3gen.safetensors — 0.1 / 1.1 GB",
        "Generating step 1/3.",
        "Generating step 2/3.",
        "Generating step 3/3.",
        "Decoding.",
        "Finishing output.",
        "Loading perth implicit.safetensors — 0.0 / 37.4 MB",
        "Finishing output.",
        "Saving audio.",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    validate_progress(&expected, "audio").unwrap();
    let coalesced_reader: Vec<_> = expected
        .iter()
        .filter(|m| m.as_str() != "Loading s3gen.safetensors — 0.0 / 1.1 GB")
        .cloned()
        .collect();
    validate_progress(&coalesced_reader, "audio").unwrap();
    // Brief compute markers need not survive the HTTP coalescer. The later PerTh
    // IO boundary still forbids restoration of a completed renderer step.
    let coalesced: Vec<_> = expected
        .iter()
        .filter(|m| !matches!(m.as_str(), "Decoding." | "Finishing output."))
        .cloned()
        .collect();
    validate_progress(&coalesced, "audio").unwrap();
    for (position, obsolete) in [
        (7, "Encoding prompt."),
        (9, "Generating step 63/4096."),
        (12, "Loading s3gen.safetensors — 0.2 / 1.1 GB"),
        (14, "Generating step 2/3."),
        (9, "Decoding."), // Premature decode cannot precede renderer IO or stages.
        (16, "Generating step 3/3."), // No stage may return after actual HiFT decode.
        (17, "Generating step 3/3."),
        (18, "Preparing audio."),
        (7, "Loading t3 cfg.safetensors — 0.2 / 2.1 GB"),
        (3, "Loading t3 cfg.safetensors — 0.0 / 2.1 GB"),
    ] {
        let mut stale = expected.clone();
        stale.insert(position, obsolete.into());
        assert!(
            validate_progress(&stale, "audio").is_err(),
            "{obsolete} at {position}"
        );
    }
    // Retirement follows any observed later phase, even when earlier callbacks
    // were coalesced. No omitted marker is required to appear on the wire.
    let omitted_prompt: Vec<_> = expected
        .iter()
        .filter(|m| m.as_str() != "Encoding prompt.")
        .cloned()
        .collect();
    validate_progress(&omitted_prompt, "audio").unwrap();
    let mut late_conditioning = omitted_prompt;
    let ar = late_conditioning
        .iter()
        .position(|m| m.contains("13/4096"))
        .unwrap();
    late_conditioning.insert(ar + 1, "Loading t3 cfg.safetensors — 0.2 / 2.1 GB".into());
    assert!(validate_progress(&late_conditioning, "audio").is_err());
    let omitted_ar: Vec<_> = expected
        .iter()
        .filter(|m| !m.contains("/4096"))
        .cloned()
        .collect();
    validate_progress(&omitted_ar, "audio").unwrap();
    let renderer = omitted_ar
        .iter()
        .position(|m| m.contains("0.3 / 1.1"))
        .unwrap();
    for obsolete in ["Encoding prompt.", "Generating step 63/4096."] {
        let mut stale = omitted_ar.clone();
        stale.insert(renderer + 1, obsolete.into());
        assert!(validate_progress(&stale, "audio").is_err(), "{obsolete}");
    }
    for later in [
        "Generating step 1/3.",
        "Decoding.",
        "Finishing output.",
        "Loading perth implicit.safetensors — 0.0 / 37.4 MB",
    ] {
        let mut omitted_early: Vec<_> = expected
            .iter()
            .filter(|m| m.as_str() != "Encoding prompt." && !m.contains("/4096"))
            .cloned()
            .collect();
        validate_progress(&omitted_early, "audio").unwrap();
        let boundary = omitted_early
            .iter()
            .position(|m| m.as_str() == later)
            .unwrap();
        omitted_early.insert(boundary + 1, "Synthesizing audio.".into());
        assert!(
            validate_progress(&omitted_early, "audio").is_err(),
            "{later}"
        );
    }
    for later in [
        "Finishing output.",
        "Loading perth implicit.safetensors — 0.0 / 37.4 MB",
        "Saving audio.",
    ] {
        let mut stale_decode = expected.clone();
        let boundary = stale_decode
            .iter()
            .position(|m| m.as_str() == later)
            .unwrap();
        stale_decode.insert(boundary + 1, "Decoding.".into());
        assert_eq!(
            validate_progress(&stale_decode, "audio").unwrap_err(),
            "audio decoding returned after postprocessing began"
        );
    }
    let mut stale = coalesced;
    let perth = stale
        .iter()
        .position(|m| m.contains("perth implicit"))
        .unwrap();
    stale.insert(perth + 1, "Generating step 3/3.".into());
    assert_eq!(
        validate_progress(&stale, "audio").unwrap_err(),
        "audio step returned after output finishing began"
    );
}

#[test]
fn generated_asset_paths_follow_the_typed_image_and_audio_file_contract() {
    use sceneworks_core::contracts::AssetFile;

    for (path, mime_type) in [
        ("assets/images/genset/image.png", "image/png"),
        ("assets/audios/genset/audio.wav", "audio/wav"),
    ] {
        let file = AssetFile {
            path: path.into(),
            mime_type: mime_type.into(),
            width: None,
            height: None,
            duration: None,
            fps: None,
            extra: Default::default(),
        };
        let asset = json!({"file":file,"relativePath":"obsolete-top-level-path"});
        assert_eq!(asset_file_path(&asset).unwrap(), path);
    }
    for asset in [
        json!({"relativePath":"assets/images/legacy.png"}),
        json!({"file":{},"relativePath":"assets/audios/legacy.wav"}),
        json!({"file":{"path":""}}),
        json!({"file":{"path":null}}),
    ] {
        assert!(asset_file_path(&asset).is_err());
    }
}

#[test]
fn training_export_collector_accepts_normal_completion_and_requires_the_actual_export() {
    use sceneworks_core::contracts::JobStatus;

    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("adapter.safetensors");
    let bytes = b"declared opaque export fixture";
    std::fs::write(&output, bytes).unwrap();
    // Normal training_result fields; trainedArtifactExported is only a canceled-export marker.
    let completed = json!({
        "status":JobStatus::Completed,
        "result":{"mode":"train","loraId":"load-progress-one-update","backend":"candle",
                  "steps":1,"stepsCompleted":1,"outputPath":output}
    });
    let evidence = training_export_evidence(&completed).unwrap();
    assert_eq!(evidence["path"], json!(output));
    assert_eq!(evidence["bytes"], bytes.len());
    assert_eq!(evidence["sha256"], sha(bytes));
    assert_eq!(evidence["loraId"], completed["result"]["loraId"]);

    let mut canceled = completed.clone();
    canceled["status"] = json!(JobStatus::Canceled);
    canceled["result"]["trainedArtifactExported"] = json!(true);
    assert!(training_export_evidence(&canceled).is_err());
    let mut no_step = completed.clone();
    no_step["result"]["stepsCompleted"] = json!(0);
    assert!(training_export_evidence(&no_step).is_err());
    let mut missing_path = completed.clone();
    missing_path["result"]
        .as_object_mut()
        .unwrap()
        .remove("outputPath");
    assert!(training_export_evidence(&missing_path).is_err());
    std::fs::write(&output, []).unwrap();
    assert!(training_export_evidence(&completed).is_err());
    std::fs::remove_file(&output).unwrap();
    assert!(training_export_evidence(&completed).is_err());
}
