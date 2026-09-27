//! YuE2 job tests (sc-22999): the real job path — eligibility, resolution, request mapping,
//! progress, supervision, cancel, resume, publication — driven against a stub API and a stub
//! [`Generator`] that behaves like the `yue2` provider's `generate_with_report` (it publishes a
//! run directory the engine's own `verify_run` accepts).

use super::*;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode as HttpStatus;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

const BF16_REVISION: &str = "1a96eca688d6ae5d7f0feb88573fec89920fcd19";
const VAE_REVISION: &str = "95535e72a97bc0f09b8ada125d26b4009428c0e8";
const VAE_LEGACY_REVISION: &str = "b54118f0fc462f08999d1ec07e88817f4ee3f770";
const SCORE: &str = include_str!("../../../sceneworks-core/src/yue2_score/fixtures/score.abc");

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
        .unwrap()
        .iter()
        .find(|model| model["id"] == "yue2")
        .expect("yue2 is in the builtin catalog")
        .clone()
}

// ---------------------------------------------------------------------------------------------
// Stub API.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Default)]
struct ApiState {
    /// `None` ⇒ eligible; `Some((status, code))` ⇒ refused.
    refusal: Arc<Mutex<Option<(u16, &'static str)>>>,
    cancel_requested: Arc<AtomicBool>,
    progress: Arc<Mutex<Vec<Value>>>,
    heartbeats: Arc<AtomicUsize>,
    eligibility_calls: Arc<AtomicUsize>,
    versions: Arc<Mutex<BTreeMap<String, Value>>>,
}

fn job_json(id: &str, cancel: bool) -> Value {
    json!({
        "id": id, "type": "audio_generate", "status": "running",
        "projectId": null, "projectName": null, "payload": {}, "result": {},
        "requestedGpu": "auto", "assignedGpu": null, "workerId": "test-worker", "progress": 0,
        "stage": "running", "message": "", "error": null, "etaSeconds": null,
        "elapsedSeconds": null, "attempts": 1, "sourceJobId": null, "duplicateOfJobId": null,
        "cancelRequested": cancel, "createdAt": "2026-09-26T00:00:00Z",
        "updatedAt": "2026-09-26T00:00:00Z", "startedAt": null, "completedAt": null,
        "canceledAt": null, "lastHeartbeatAt": null
    })
}

async fn spawn_api(state: ApiState) -> String {
    async fn job(State(s): State<ApiState>, AxumPath(id): AxumPath<String>) -> Response {
        Json(job_json(&id, s.cancel_requested.load(Ordering::SeqCst))).into_response()
    }
    async fn progress(
        State(s): State<ApiState>,
        AxumPath(id): AxumPath<String>,
        Json(body): Json<Value>,
    ) -> Response {
        s.progress.lock().unwrap().push(body);
        Json(job_json(&id, s.cancel_requested.load(Ordering::SeqCst))).into_response()
    }
    async fn heartbeat(State(s): State<ApiState>) -> Response {
        s.heartbeats.fetch_add(1, Ordering::SeqCst);
        Json(json!({})).into_response()
    }
    async fn eligibility(State(s): State<ApiState>) -> Response {
        s.eligibility_calls.fetch_add(1, Ordering::SeqCst);
        match *s.refusal.lock().unwrap() {
            None => Json(json!({"eligible": true, "usagePolicy": {"modelId": "yue2", "nonCommercial": true, "checkedAt": "execution"}})).into_response(),
            Some((status, code)) => (
                HttpStatus::from_u16(status).unwrap(),
                Json(json!({"detail": format!("refused: {code}"), "code": code})),
            )
                .into_response(),
        }
    }
    async fn version(
        State(s): State<ApiState>,
        AxumPath((_project, id)): AxumPath<(String, String)>,
    ) -> Response {
        match s.versions.lock().unwrap().get(&id) {
            Some(version) => Json(json!({"version": version, "renders": []})).into_response(),
            None => (HttpStatus::NOT_FOUND, Json(json!({"detail": "no version"}))).into_response(),
        }
    }
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job))
        .route("/api/v1/jobs/:job_id/progress", post(progress))
        .route("/api/v1/jobs/:job_id/yue2-eligibility", get(eligibility))
        .route("/api/v1/workers/:worker_id/heartbeat", post(heartbeat))
        .route(
            "/api/v1/projects/:project_id/yue2/score-versions/:version_id",
            get(version),
        )
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{address}")
}

// ---------------------------------------------------------------------------------------------
// A published run the engine's `verify_run` accepts.
// ---------------------------------------------------------------------------------------------

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> Value {
    std::fs::write(dir.join(name), bytes).unwrap();
    json!({"sha256": sha256_hex(bytes), "bytes": bytes.len()})
}

struct FakeRun<'a> {
    kind: &'a str,
    identity: &'a str,
    plan_identity: &'a str,
    truncated: (bool, bool),
    request: Value,
    score: Option<&'a str>,
}

/// Publish a run directory at `dir` with the files `verify_run` requires for its kind.
fn publish_fake_run(dir: &Path, run: &FakeRun) {
    std::fs::create_dir_all(dir).unwrap();
    let mut artifacts = serde_json::Map::new();
    let mut add = |name: &str, bytes: &[u8]| {
        artifacts.insert(name.to_owned(), write_file(dir, name, bytes));
    };
    for name in [
        "plan.json",
        "abc_tokens.npy",
        "prefix.npy",
        "plan_manifest.json",
    ] {
        add(name, format!("{name} of {}", run.identity).as_bytes());
    }
    if let Some(score) = run.score {
        add("score.abc", score.as_bytes());
    }
    add(
        "request.json",
        serde_json::to_vec(&run.request).unwrap().as_slice(),
    );
    if run.kind == "plan" {
        add(
            "provenance.json",
            serde_json::to_vec(&json!({"config": {"engine": "stub", "plan_only": true}}))
                .unwrap()
                .as_slice(),
        );
    } else {
        for name in ["semantic.npy", "latent.npy", "latent.json", "audio.wav"] {
            add(name, format!("{name} of {}", run.identity).as_bytes());
        }
        add(
            "config.json",
            serde_json::to_vec(&json!({"engine": "stub", "ode_steps": 8}))
                .unwrap()
                .as_slice(),
        );
    }
    let result = json!({
        "schema": "yue2-run-v1",
        "kind": run.kind,
        "status": "complete",
        "identity": run.identity,
        "plan_identity": run.plan_identity,
        "truncated": {"abc": run.truncated.0, "semantic": run.truncated.1},
        "weights": {"mot": "stub-mot"},
        "license": {"weights": "cc-by-nc-4.0"},
        "artifacts": artifacts,
    });
    std::fs::write(
        dir.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
}

// ---------------------------------------------------------------------------------------------
// Stub generator.
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
enum Behavior {
    /// Emit `steps`, then publish a run of `kind` (truncated as given) and return its report.
    Complete {
        steps: Vec<Progress>,
        truncated: (bool, bool),
    },
    /// Emit many steps for about `millis` without yielding to the runtime, then complete.
    LongAr { steps: u32, millis: u64 },
    /// Open the run's working directory (with the engine's claim) and wait for `req.cancel`.
    /// (Its test decodes a source run, which only a build with the audio lane can verify.)
    #[cfg_attr(
        not(any(target_os = "macos", feature = "backend-candle")),
        allow(dead_code)
    )]
    WaitForCancel,
    /// Leave a working directory with a checkpoint, then fail.
    Fail,
}

#[derive(Default)]
struct Seen {
    load: Option<LoadSpec>,
    request: Option<GenerationRequest>,
    observed_cancel: bool,
}

struct StubYue2 {
    descriptor: gen_core::ModelDescriptor,
    behavior: Behavior,
    seen: Arc<Mutex<Seen>>,
}

fn stub_descriptor() -> gen_core::ModelDescriptor {
    gen_core::ModelDescriptor {
        id: "yue2",
        family: "yue2",
        backend: "candle",
        modality: gen_core::Modality::Audio,
        capabilities: gen_core::Capabilities {
            supports_symbolic_song: true,
            supports_audio_artifacts: true,
            supports_song_plan_only: true,
            supports_song_cover: true,
            ..Default::default()
        },
        encoder_contract: None,
        denoiser_output_latent_space: None,
        required_components: &[],
        control_kinds: None,
    }
}

impl StubYue2 {
    fn kind_of(req: &GenerationRequest) -> &'static str {
        let song = req.audio.as_ref().and_then(|a| a.song.as_ref());
        match song {
            Some(s) if s.plan_only => "plan",
            Some(s) if s.cached_latents.is_some() => "cached_decode",
            _ => "song",
        }
    }

    fn publish(
        req: &GenerationRequest,
        truncated: (bool, bool),
    ) -> gen_core::Result<GenerationReport> {
        let artifacts = req
            .audio
            .as_ref()
            .and_then(|a| a.artifacts.clone())
            .unwrap();
        let kind = Self::kind_of(req);
        let identity = format!("{:x}", Sha256::digest(format!("{kind}:{}", req.prompt)));
        let partial = partial_dir(&artifacts.dir);
        let _ = std::fs::remove_dir_all(&partial);
        let request = json!({
            "style": req.prompt,
            "lyrics": req.audio.as_ref().and_then(|a| a.lyrics.clone()).unwrap_or_default(),
            "cot": match req.audio.as_ref().and_then(|a| a.song.as_ref()).and_then(|s| s.planning) {
                Some(SongPlanning::Melody) => "melody",
                Some(SongPlanning::Off) => "off",
                _ => "full",
            },
            "seed": req.seed.unwrap_or(831_001),
            "cfg_scale": req.guidance.map(f64::from),
            "id": "song",
        });
        publish_fake_run(
            &artifacts.dir,
            &FakeRun {
                kind,
                identity: &identity,
                plan_identity: "ab12",
                truncated,
                request,
                score: Some(SCORE),
            },
        );
        let mut warnings = Vec::new();
        if truncated.0 {
            warnings.push(gen_core::GenerationWarning {
                code: "abc_truncated".into(),
                message: "the ABC score phase hit its max_tokens budget".into(),
            });
        }
        if truncated.1 {
            warnings.push(gen_core::GenerationWarning {
                code: "semantic_truncated".into(),
                message: "the semantic phase hit its max_tokens budget".into(),
            });
        }
        Ok(GenerationReport {
            output: (kind != "plan").then(|| {
                GenerationOutput::Audio(gen_core::AudioTrack {
                    samples: vec![0.2, -0.2, 0.1, -0.1, 0.3, -0.3],
                    sample_rate: 48_000,
                    channels: 2,
                    stems: Vec::new(),
                })
            }),
            artifacts: Some(gen_core::ArtifactRecord {
                dir: artifacts.dir.clone(),
                kind: kind.to_owned(),
                identity,
            }),
            warnings,
        })
    }
}

impl Generator for StubYue2 {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }
    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        Err(gen_core::Error::Msg(
            "the YuE2 job must call generate_with_report".into(),
        ))
    }
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        self.seen.lock().unwrap().request = Some(req.clone());
        let artifacts = req
            .audio
            .as_ref()
            .and_then(|a| a.artifacts.clone())
            .unwrap();
        match &self.behavior {
            Behavior::Complete { steps, truncated } => {
                for step in steps {
                    on_progress(*step);
                }
                Self::publish(req, *truncated)
            }
            Behavior::LongAr { steps, millis } => {
                let pause = Duration::from_micros(millis * 1000 / u64::from(*steps));
                for current in 1..=*steps {
                    on_progress(Progress::Step {
                        current,
                        total: *steps,
                    });
                    std::thread::sleep(pause);
                }
                Self::publish(req, (false, false))
            }
            Behavior::WaitForCancel => {
                let work = partial_dir(&artifacts.dir);
                std::fs::create_dir_all(&work).unwrap();
                std::fs::write(
                    work.join(RUN_LOCK_FILE),
                    format!("pid {}\n", std::process::id()),
                )
                .unwrap();
                std::fs::write(work.join("plan.json"), b"checkpoint").unwrap();
                let start = Instant::now();
                while !req.cancel.is_cancelled() {
                    on_progress(Progress::Step {
                        current: 1,
                        total: 4096,
                    });
                    if start.elapsed() > Duration::from_secs(30) {
                        return Err(gen_core::Error::Msg("req.cancel never tripped".into()));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                // The engine's own hook saw the cancel (not a job flag the worker reads).
                self.seen.lock().unwrap().observed_cancel = true;
                // Like the engine, the claim is released when the run is cancelled.
                let _ = std::fs::remove_file(work.join(RUN_LOCK_FILE));
                Err(gen_core::Error::Canceled)
            }
            Behavior::Fail => {
                let work = partial_dir(&artifacts.dir);
                std::fs::create_dir_all(&work).unwrap();
                std::fs::write(work.join("plan.json"), b"checkpoint").unwrap();
                Err(gen_core::Error::Msg("semantic stage failed".into()))
            }
        }
    }
}

type Loader = Box<dyn FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send>;

fn loader(behavior: Behavior, seen: Arc<Mutex<Seen>>, loads: Arc<AtomicUsize>) -> Loader {
    Box::new(move |id: &str, spec: &LoadSpec| {
        assert_eq!(id, "yue2", "the job loads the registered yue2 provider");
        loads.fetch_add(1, Ordering::SeqCst);
        seen.lock().unwrap().load = Some(spec.clone());
        Ok(Box::new(StubYue2 {
            descriptor: stub_descriptor(),
            behavior,
            seen,
        }) as Box<dyn Generator>)
    })
}

// ---------------------------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------------------------

struct Harness {
    _env: crate::test_env::EnvVars,
    data: tempfile::TempDir,
    state: ApiState,
    settings: Settings,
    api: ApiClient,
    project_id: String,
    project_path: PathBuf,
    entry: Value,
}

fn stage_snapshot(data_dir: &Path, repo: &str, revision: &str) -> PathBuf {
    let dir = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
    dir
}

impl Harness {
    async fn new() -> Self {
        Self::with_decoders(true, true).await
    }

    async fn with_decoders(standard: bool, legacy: bool) -> Self {
        let env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ]);
        let data = tempfile::tempdir().unwrap();
        stage_snapshot(data.path(), "m-a-p/YuE2-3B", BF16_REVISION);
        if standard {
            stage_snapshot(data.path(), "m-a-p/YuE2-Vae", VAE_REVISION);
        }
        if legacy {
            stage_snapshot(data.path(), "m-a-p/YuE2-Vae-legacy", VAE_LEGACY_REVISION);
        }
        Self::finish(env, data).await
    }

    /// A harness over a real Hugging Face hub holding the pinned YuE2 snapshots.
    #[cfg(target_os = "macos")]
    async fn over_hub(hub: &str) -> Self {
        let env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ]);
        let data = tempfile::tempdir().unwrap();
        Self::finish(env, data).await
    }

    async fn finish(env: crate::test_env::EnvVars, data: tempfile::TempDir) -> Self {
        let project = ProjectStore::new(data.path().to_path_buf(), "test")
            .create_project("YuE2 songs")
            .unwrap();
        let state = ApiState::default();
        let url = spawn_api(state.clone()).await;
        let mut settings = Settings::from_env();
        settings.api_url = url;
        settings.worker_id = "test-worker".into();
        settings.data_dir = data.path().to_path_buf();
        settings.heartbeat_seconds = 5;
        let api = ApiClient::new(&settings);
        Self {
            _env: env,
            state,
            api,
            settings,
            project_id: project.id,
            project_path: PathBuf::from(project.path),
            data,
            entry: builtin_yue2(),
        }
    }

    fn job(&self, id: &str, mut spec: Value) -> JobSnapshot {
        if spec.get("runId").is_none() {
            spec["runId"] = json!(format!("yue2run_{id}"));
        }
        let parsed: Yue2JobSpec = serde_json::from_value(spec.clone()).unwrap();
        contract::validate_for_execution(&parsed).unwrap_or_else(|e| panic!("fixture: {e}"));
        let mut job: JobSnapshot = serde_json::from_value(job_json(id, false)).unwrap();
        job.project_id = Some(self.project_id.clone());
        job.payload = json!({
            "projectId": self.project_id,
            "model": "yue2",
            "modelManifestEntry": self.entry,
            "yue2": spec,
            "usagePolicy": {"modelId": "yue2", "checkedAt": "submission"},
            "commercialUse": false,
        })
        .as_object()
        .unwrap()
        .clone();
        job
    }

    async fn run(&self, job: &JobSnapshot, loader: Loader) -> WorkerResult<()> {
        run_yue2_job_using(&self.api, &self.settings, job, loader).await
    }

    fn progress(&self) -> Vec<Value> {
        self.state.progress.lock().unwrap().clone()
    }

    fn terminal(&self) -> Value {
        self.progress()
            .into_iter()
            .rev()
            .find(|p| {
                matches!(
                    p["status"].as_str(),
                    Some("completed" | "failed" | "canceled")
                )
            })
            .expect("a terminal status was posted")
    }

    fn run_dir(&self, run_id: &str) -> PathBuf {
        self.project_path.join(contract::run_dir(run_id))
    }
}

fn complete(steps: Vec<Progress>) -> Behavior {
    Behavior::Complete {
        steps,
        truncated: (false, false),
    }
}

#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
fn song_request_json(style: &str) -> Value {
    json!({"style": style, "lyrics": "[verse]\nla", "cot": "full", "seed": 3, "id": "song"})
}

// ---------------------------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------------------------

/// A create job reaches the provider with EVERY applicable setting mapped, through
/// `generate_with_report`, with a stable resumable run directory; the result persists the run
/// record, the effective settings and the policy, and the audio lands as a library asset.
#[tokio::test]
async fn a_create_job_maps_every_setting_and_publishes_a_library_asset() {
    let h = Harness::new().await;
    let job = h.job(
        "create-1",
        json!({
            "kind": "create", "style": "dream pop", "lyrics": "[verse]\nla la", "seed": 42,
            "cfgScale": 1.5, "steps": 12, "planning": "melody",
            "scoreSampling": {"temperature": 0.6, "topP": 0.8, "topK": 20, "repetitionPenalty": 1.1,
                              "penaltyWindow": 64, "minTokens": 16, "maxTokens": 512},
            "semanticSampling": {"maxTokens": 300, "minTokens": 10},
            "decoder": "legacy", "precision": "fp32", "offloadPolicy": "sequential",
            "memory": {"chunkAttention": true, "attentionChunkSize": 65536,
                       "tileVaeDecode": true, "decodeTileEdge": 64},
        }),
    );
    let seen = Arc::new(Mutex::new(Seen::default()));
    let loads = Arc::new(AtomicUsize::new(0));
    h.run(
        &job,
        loader(
            complete(vec![
                Progress::Step {
                    current: 1,
                    total: 512,
                },
                Progress::Step {
                    current: 1,
                    total: 300,
                },
                Progress::Decoding,
            ]),
            seen.clone(),
            loads,
        ),
    )
    .await
    .expect("the job completes");

    let seen = seen.lock().unwrap();
    let req = seen.request.as_ref().expect("generate_with_report ran");
    assert_eq!(req.prompt, "dream pop");
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.guidance, Some(1.5));
    assert_eq!(req.steps, Some(12));
    let audio = req.audio.as_ref().unwrap();
    assert_eq!(audio.lyrics.as_deref(), Some("[verse]\nla la"));
    let song = audio.song.as_ref().unwrap();
    assert_eq!(song.planning, Some(SongPlanning::Melody));
    assert_eq!(song.decoder, Some(SongDecoder::Legacy));
    assert!(!song.plan_only);
    let score = song.score_sampling.unwrap();
    assert_eq!(
        (
            score.temperature,
            score.top_p,
            score.top_k,
            score.repetition_penalty
        ),
        (Some(0.6), Some(0.8), Some(20), Some(1.1))
    );
    assert_eq!(
        (score.penalty_window, score.min_tokens, score.max_tokens),
        (Some(64), Some(16), Some(512))
    );
    let semantic = song.semantic_sampling.unwrap();
    assert_eq!(
        (semantic.min_tokens, semantic.max_tokens),
        (Some(10), Some(300))
    );
    let artifacts = audio.artifacts.as_ref().unwrap();
    assert_eq!(artifacts.dir, h.run_dir("yue2run_create-1"));
    assert!(artifacts.resume, "a job always resumes its own run id");
    let memory = req.memory.expect("memory controls pass through");
    assert!(memory.chunk_attention && memory.tile_vae_decode);
    assert_eq!(memory.attention_chunk_size, Some(65536));
    assert_eq!(memory.decode_tile_edge, Some(64));
    // Unset stageResidency follows the load's sequential offload instead of forcing residency.
    // Mutation that reds this: `stage_residency: m.stage_residency.unwrap_or(false)`.
    assert!(memory.stage_residency);

    let load = seen.load.as_ref().unwrap();
    assert!(matches!(&load.weights, WeightsSource::Dir(dir) if dir.ends_with(BF16_REVISION)));
    assert_eq!(load.quantize, None, "bf16 is the unquantized original");
    assert_eq!(load.precision, Precision::Fp32);
    assert_eq!(load.offload_policy, GenOffloadPolicy::Sequential);
    assert_eq!(
        load.components.keys().cloned().collect::<Vec<_>>(),
        vec!["vae".to_owned(), "vae_legacy".to_owned()],
        "every installed decoder is staged"
    );

    let terminal = h.terminal();
    assert_eq!(terminal["status"], "completed", "{terminal}");
    let result = &terminal["result"];
    let block = &result["yue2"];
    assert_eq!(block["kind"], "create");
    assert_eq!(block["run"]["dir"], "yue2/runs/yue2run_create-1");
    assert_eq!(block["run"]["kind"], "song");
    assert_eq!(block["run"]["planIdentity"], "ab12");
    assert_eq!(block["effectiveSettings"]["tier"], "bf16");
    assert_eq!(block["effectiveSettings"]["decoder"], "legacy");
    assert_eq!(block["effectiveSettings"]["engineConfig"]["ode_steps"], 8);
    assert_eq!(block["decoder"]["id"], "m-a-p/YuE2-Vae-legacy");
    assert_eq!(block["model"]["id"], "m-a-p/YuE2-3B");
    // Execution-time policy, not the one stamped at submission.
    assert_eq!(block["usagePolicy"]["checkedAt"], "execution");
    assert_eq!(
        block["score"]["abc"], SCORE,
        "a planned song carries its score to version"
    );
    assert_eq!(
        block["request"]["cot"], "melody",
        "the run request in sc-22997 shape"
    );

    let fact = &result["assetWrites"][0];
    assert_eq!(fact["type"], "audio");
    assert_eq!(fact["sampleRate"], 48_000);
    assert_eq!(fact["channels"], 2);
    assert_eq!(fact["model"], "yue2");
    assert_eq!(fact["extra"]["usagePolicy"]["nonCommercial"], true);
    assert_eq!(
        fact["extra"]["yue2"]["run"]["identity"],
        block["run"]["identity"]
    );
    assert!(fact["extra"]["yue2"].get("score").is_none());
    let media = h.project_path.join(fact["mediaPath"].as_str().unwrap());
    assert!(media.is_file(), "the WAV is written into the project");

    // Progress: coalesced, monotone, ending on a stage message.
    let fractions: Vec<f64> = h
        .progress()
        .iter()
        .filter(|p| p["status"] == "running")
        .map(|p| p["progress"].as_f64().unwrap())
        .collect();
    assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "{fractions:?}");
}

/// Each remaining kind reaches the provider with its own control surface.
#[tokio::test]
async fn plan_cover_render_and_restored_plan_jobs_map_their_kind() {
    // Plan-only: `plan_only` with its artifacts, no audio, no asset; the plan's score is kept.
    let h = Harness::new().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let job = h.job(
        "plan-1",
        json!({"kind": "plan", "style": "folk", "lyrics": "[verse]\nhey", "seed": 7, "tier": "bf16"}),
    );
    h.run(
        &job,
        loader(complete(vec![]), seen.clone(), Default::default()),
    )
    .await
    .unwrap();
    {
        let seen = seen.lock().unwrap();
        let song = seen
            .request
            .as_ref()
            .unwrap()
            .audio
            .as_ref()
            .unwrap()
            .song
            .clone()
            .unwrap();
        assert!(song.plan_only);
        assert_eq!(song.decoder, None, "a plan-only job renders no audio");
        assert_eq!(song.semantic_sampling, None);
    }
    let result = h.terminal()["result"].clone();
    assert!(result.get("assetWrites").is_none(), "{result}");
    assert_eq!(result["yue2"]["run"]["kind"], "plan");
    assert_eq!(result["yue2"]["score"]["abc"], SCORE);

    // Cover from a reviewed score. (One harness at a time: each holds the crate's env lock.)
    drop(h);
    let h = Harness::new().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let job = h.job(
        "cover-1",
        json!({"kind": "cover", "style": "bossa", "lyrics": "[verse]\nola", "seed": 5,
               "cover": {"mode": "melody", "score": SCORE, "keep": "vocal", "translatedFrom": "[verse]\nhi"}}),
    );
    h.run(
        &job,
        loader(complete(vec![]), seen.clone(), Default::default()),
    )
    .await
    .unwrap();
    {
        let seen = seen.lock().unwrap();
        let cover = seen
            .request
            .as_ref()
            .unwrap()
            .audio
            .as_ref()
            .unwrap()
            .song
            .as_ref()
            .unwrap()
            .cover
            .clone()
            .unwrap();
        assert_eq!(cover.mode, SongCoverMode::Melody);
        assert_eq!(cover.keep, Some(SongCoverVoice::Vocal));
        assert_eq!(cover.score, SCORE);
        assert_eq!(cover.translated_from.as_deref(), Some("[verse]\nhi"));
    }
    assert_eq!(h.terminal()["status"], "completed");

    // A score-version render: the version's request and ABC, read at execution.
    drop(h);
    let h = Harness::new().await;
    let version = version_fixture("yue2v_aaaa");
    h.state
        .versions
        .lock()
        .unwrap()
        .insert("yue2v_aaaa".into(), version.clone());
    let seen = Arc::new(Mutex::new(Seen::default()));
    let job = h.job("render-1", render_spec(&version, None));
    h.run(
        &job,
        loader(complete(vec![]), seen.clone(), Default::default()),
    )
    .await
    .unwrap();
    {
        let seen = seen.lock().unwrap();
        let req = seen.request.as_ref().unwrap();
        assert_eq!(req.prompt, "city pop");
        assert_eq!(req.seed, Some(9));
        assert_eq!(req.guidance, Some(2.0));
        let song = req.audio.as_ref().unwrap().song.as_ref().unwrap();
        assert_eq!(song.planning, Some(SongPlanning::Full));
        assert_eq!(song.score.as_deref(), Some(SCORE));
    }
    let block = h.terminal()["result"]["yue2"].clone();
    assert_eq!(block["versionId"], "yue2v_aaaa");
    assert_eq!(block["renderedScoreSha256"], version["score"]["sha256"]);
    assert_eq!(block["renderedRequestSha256"], version["requestSha256"]);
    assert!(
        block.get("score").is_none(),
        "a render versions nothing new"
    );
}

fn version_fixture(id: &str) -> Value {
    let request = sceneworks_core::yue2_score::SongRequest {
        style: "city pop".into(),
        lyrics: "[verse]\nneon".into(),
        cot: sceneworks_core::yue2_score::Cot::Full,
        seed: 9,
        cfg_scale: Some(2.0),
    };
    json!({
        "schema": "sceneworks.yue2.scoreVersion.v1", "id": id, "projectId": "p", "engine": "yue2",
        "createdAt": "2026-09-26T00:00:00Z", "parentVersionId": null, "rootVersionId": id,
        "origin": "import", "request": request,
        "requestSha256": sceneworks_core::yue2_score::request_sha256(&request),
        "score": {"abc": SCORE, "sha256": sha256_hex(SCORE.as_bytes()), "summary": {}},
        "edit": null, "provenance": {"actor": "user", "channel": "api"}, "renderNotice": "",
    })
}

fn render_spec(version: &Value, score_sha: Option<&str>) -> Value {
    json!({
        "kind": "renderVersion", "versionId": version["id"], "steps": 6,
        "semanticSampling": {"temperature": 0.9}, "decoder": "standard",
        "sources": {"version": {
            "id": version["id"],
            "scoreSha256": score_sha.unwrap_or(version["score"]["sha256"].as_str().unwrap()),
            "requestSha256": version["requestSha256"],
        }},
    })
}

/// A score version whose digests are not the ones the job was queued with is refused before load.
#[tokio::test]
async fn a_render_of_a_changed_score_version_is_refused_before_load() {
    let h = Harness::new().await;
    let version = version_fixture("yue2v_bbbb");
    h.state
        .versions
        .lock()
        .unwrap()
        .insert("yue2v_bbbb".into(), version.clone());
    let loads = Arc::new(AtomicUsize::new(0));
    let job = h.job("render-2", render_spec(&version, Some(&"0".repeat(64))));
    h.run(
        &job,
        loader(complete(vec![]), Default::default(), loads.clone()),
    )
    .await
    .unwrap();
    // Mutation that reds this: dropping the digest comparison in `resolve_inputs`.
    assert_eq!(loads.load(Ordering::SeqCst), 0);
    let terminal = h.terminal();
    assert_eq!(terminal["status"], "failed");
    assert!(
        terminal["error"].as_str().unwrap().contains("digests"),
        "{terminal}"
    );
}

/// Eligibility is enforced AT EXECUTION: the job was queued under an acknowledgment (its payload
/// says so), the acknowledgment is withdrawn before the worker runs it, and the job is refused
/// before any weight loads. The same for a commercial-use refusal.
#[tokio::test]
async fn withdrawn_eligibility_refuses_the_job_at_execution_before_load() {
    for code in ["license_acknowledgment_required", "commercial_use_refused"] {
        let h = Harness::new().await;
        let job = h.job(
            "elig-1",
            json!({"kind": "create", "style": "x", "lyrics": "[verse]\nla"}),
        );
        // The state changes between submission and execution.
        *h.state.refusal.lock().unwrap() = Some((403, code));
        let loads = Arc::new(AtomicUsize::new(0));
        h.run(
            &job,
            loader(complete(vec![]), Default::default(), loads.clone()),
        )
        .await
        .unwrap();
        assert_eq!(h.state.eligibility_calls.load(Ordering::SeqCst), 1);
        // Mutation that reds this: skipping `check_eligibility` (the loader then runs).
        assert_eq!(loads.load(Ordering::SeqCst), 0, "{code}: nothing loads");
        let terminal = h.terminal();
        assert_eq!(terminal["status"], "failed");
        assert!(
            terminal["error"].as_str().unwrap().contains(code),
            "{terminal}"
        );
        // The failure keeps the submission-time policy it was queued under.
        assert_eq!(
            terminal["result"]["yue2"]["usagePolicy"]["checkedAt"],
            "submission"
        );
        assert!(
            !h.run_dir("yue2run_elig-1").exists()
                && !partial_dir(&h.run_dir("yue2run_elig-1")).exists()
        );
    }
}

/// A cancel reaches the engine's own hook (`GenerationRequest::cancel`), the job reads canceled,
/// and exactly the run's `.partial` working directory is removed — a decode's source run, which
/// the job only reads, is untouched.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
#[tokio::test]
async fn a_cancel_reaches_the_engine_hook_and_removes_only_the_runs_working_directory() {
    let h = Harness::new().await;
    let source = h.run_dir("yue2run_source");
    publish_fake_run(
        &source,
        &FakeRun {
            kind: "song",
            identity: "cafe01",
            plan_identity: "ab12",
            truncated: (false, false),
            request: song_request_json("x"),
            score: None,
        },
    );
    let source_bytes = std::fs::read(source.join("result.json")).unwrap();
    let job = h.job(
        "cancel-1",
        json!({"kind": "decode", "sourceJobId": "job-src", "decoder": "standard",
               "sources": {"sourceRun": {"jobId": "job-src", "runDir": "yue2/runs/yue2run_source",
                                          "identity": "cafe01"}}}),
    );
    let seen = Arc::new(Mutex::new(Seen::default()));
    let cancel = h.state.cancel_requested.clone();
    let partial = partial_dir(&h.run_dir("yue2run_cancel-1"));
    let watcher = {
        let partial = partial.clone();
        tokio::spawn(async move {
            // Cancel once the engine is inside its run (its working directory exists).
            while !partial.is_dir() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            cancel.store(true, Ordering::SeqCst);
        })
    };
    let outcome = h
        .run(
            &job,
            loader(Behavior::WaitForCancel, seen.clone(), Default::default()),
        )
        .await;
    watcher.await.unwrap();
    assert!(
        matches!(outcome, Err(WorkerError::Canceled(_))),
        "{outcome:?}"
    );
    // Asserted on the engine's hook, not on a job flag.
    assert!(seen.lock().unwrap().observed_cancel);
    assert_eq!(h.terminal()["status"], "canceled");
    // Mutation that reds this: dropping `remove_partial` from the canceled arm.
    assert!(!partial.exists(), "the working directory is cleaned");
    assert_eq!(
        std::fs::read(source.join("result.json")).unwrap(),
        source_bytes,
        "the source run is never touched"
    );
    // The cached decode pointed at the verified source run.
    let seen = seen.lock().unwrap();
    let song = seen
        .request
        .as_ref()
        .unwrap()
        .audio
        .as_ref()
        .unwrap()
        .song
        .clone()
        .unwrap();
    assert_eq!(song.cached_latents, Some(source));
    assert!(seen.request.as_ref().unwrap().prompt.is_empty());
}

/// A long AR phase — tens of thousands of per-token steps on a blocking thread — never starves
/// supervision: heartbeats keep flowing on the progress interval and the per-token progress is
/// coalesced into a bounded number of monotone posts.
#[tokio::test]
async fn a_long_ar_phase_keeps_heartbeats_flowing_and_coalesces_progress() {
    let h = Harness::new().await;
    let job = h.job(
        "long-1",
        json!({"kind": "create", "style": "x", "lyrics": "[verse]\nla",
               "scoreSampling": {"maxTokens": 20000}}),
    );
    let before = Instant::now();
    h.run(
        &job,
        loader(
            Behavior::LongAr {
                steps: 20_000,
                millis: 6_000,
            },
            Default::default(),
            Default::default(),
        ),
    )
    .await
    .unwrap();
    assert!(before.elapsed() >= Duration::from_secs(6));
    // The watcher's first tick is immediate; a second means the interval kept firing while the
    // blocking task held its thread. Mutation that reds this: running generation inline on the
    // async task instead of `spawn_blocking` + `run_blocking_with_heartbeat`.
    let beats = h.state.heartbeats.load(Ordering::SeqCst);
    assert!(beats >= 3, "heartbeats during a 6 s AR phase: {beats}");
    let running: Vec<Value> = h
        .progress()
        .into_iter()
        .filter(|p| {
            p["status"] == "running" && p["message"].as_str().unwrap_or("").starts_with("Planning")
        })
        .collect();
    // Mutation that reds this: posting every step (PROGRESS_POST_INTERVAL removed).
    assert!(
        (2..=40).contains(&running.len()),
        "20k steps coalesced into {} posts",
        running.len()
    );
    let fractions: Vec<f64> = running
        .iter()
        .map(|p| p["progress"].as_f64().unwrap())
        .collect();
    assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "{fractions:?}");
    assert_eq!(h.terminal()["status"], "completed");
}

#[test]
fn progress_maps_plan_semantic_acoustic_and_decode_in_order() {
    let spec: Yue2JobSpec = serde_json::from_value(json!({
        "kind": "create", "lyrics": "l", "scoreSampling": {"maxTokens": 100},
        "semanticSampling": {"maxTokens": 200}
    }))
    .unwrap();
    let mut p = Yue2Progress::new(&spec);
    let mut step = |current, total| p.observe(Progress::Step { current, total });
    let (f1, m1) = step(50, 100);
    assert!(
        m1.starts_with("Planning the score: token 50 of 100"),
        "{m1}"
    );
    let (f2, m2) = step(100, 100);
    let (f3, m3) = step(1, 200);
    assert!(
        m3.starts_with("Generating semantic tokens: 1 of 200"),
        "{m3}"
    );
    let (f4, _) = step(200, 200);
    // The acoustic stage restarts the count (its total may equal a phase's budget).
    let (f5, m5) = step(1, 200);
    assert!(m5.starts_with("Acoustic synthesis: step 1 of 200"), "{m5}");
    let (f6, m6) = p.observe(Progress::Decoding);
    assert_eq!(m6, "Decoding audio.");
    let fractions = [f1, f2, f3, f4, f5, f6];
    assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "{fractions:?}");
    assert!(f2 <= 0.30 && f4 <= 0.65 && f6 < 0.9, "{fractions:?}");
    let _ = m2;

    // A resumed run whose plan was reused starts at the semantic phase: recognised by its total.
    // Mutation that reds this: always starting at phase 0 (the plan).
    let mut resumed = Yue2Progress::new(&spec);
    let (_, m) = resumed.observe(Progress::Step {
        current: 1,
        total: 200,
    });
    assert!(m.starts_with("Generating semantic tokens"), "{m}");

    // An external score has no sampled plan phase.
    let scored: Yue2JobSpec = serde_json::from_value(json!({
        "kind": "create", "lyrics": "l", "score": SCORE
    }))
    .unwrap();
    let (_, m) = Yue2Progress::new(&scored).observe(Progress::Step {
        current: 1,
        total: 4096,
    });
    assert!(!m.starts_with("Planning"), "{m}");
}

/// Restart/resume: the job resumes its own stable run directory, a claim left by a dead worker
/// is reclaimed, and a live claim is refused (the lock is kept).
#[cfg(unix)]
#[tokio::test]
async fn a_resumed_job_reclaims_a_dead_workers_claim_and_refuses_a_live_one() {
    let h = Harness::new().await;
    let run_dir = h.run_dir("yue2run_resume-1");
    let work = partial_dir(&run_dir);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("plan.json"), b"checkpoint").unwrap();
    // A pid that is certainly gone: a child that already exited and was reaped.
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let dead = child.id();
    child.wait().unwrap();
    std::fs::write(work.join(RUN_LOCK_FILE), format!("pid {dead}\n")).unwrap();
    let seen = Arc::new(Mutex::new(Seen::default()));
    let job = h.job(
        "resume-1",
        json!({"kind": "create", "style": "x", "lyrics": "[verse]\nla"}),
    );
    h.run(
        &job,
        loader(complete(vec![]), seen.clone(), Default::default()),
    )
    .await
    .unwrap();
    assert_eq!(h.terminal()["status"], "completed");
    let artifacts = seen
        .lock()
        .unwrap()
        .request
        .as_ref()
        .unwrap()
        .audio
        .as_ref()
        .unwrap()
        .artifacts
        .clone()
        .unwrap();
    assert_eq!(artifacts.dir, run_dir);
    assert!(artifacts.resume);

    // A live claimant (this very process — e.g. an abandoned task) keeps its claim.
    drop(h);
    let h = Harness::new().await;
    let work = partial_dir(&h.run_dir("yue2run_resume-2"));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(
        work.join(RUN_LOCK_FILE),
        format!("pid {}\n", std::process::id()),
    )
    .unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let job = h.job(
        "resume-2",
        json!({"kind": "create", "style": "x", "lyrics": "[verse]\nla"}),
    );
    h.run(
        &job,
        loader(complete(vec![]), Default::default(), loads.clone()),
    )
    .await
    .unwrap();
    // Mutation that reds this: reclaiming every claim (`stale = true`).
    assert_eq!(loads.load(Ordering::SeqCst), 0);
    assert_eq!(h.terminal()["status"], "failed");
    assert!(
        work.join(RUN_LOCK_FILE).is_file(),
        "a live claim is never removed"
    );
}

/// A source run is verified against the identity its producing job recorded before anything
/// reads it: a decode of a run that is not that run, or a restore of a plan whose identity moved,
/// is refused before load.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
#[tokio::test]
async fn a_source_run_that_is_not_the_recorded_one_is_refused_before_load() {
    let h = Harness::new().await;
    publish_fake_run(
        &h.run_dir("yue2run_src"),
        &FakeRun {
            kind: "song",
            identity: "cafe01",
            plan_identity: "ab12",
            truncated: (false, false),
            request: song_request_json("x"),
            score: None,
        },
    );
    for (id, spec) in [
        (
            "decode-bad",
            json!({"kind": "decode", "sourceJobId": "job-src",
                   "sources": {"sourceRun": {"jobId": "job-src", "runDir": "yue2/runs/yue2run_src", "identity": "beef02"}}}),
        ),
        (
            "plan-bad",
            json!({"kind": "fromPlan", "planJobId": "job-src",
                   "sources": {"plan": {"jobId": "job-src", "runDir": "yue2/runs/yue2run_src",
                                         "identity": "cafe01", "planIdentity": "ff99"}}}),
        ),
    ] {
        let loads = Arc::new(AtomicUsize::new(0));
        let job = h.job(id, spec);
        h.run(
            &job,
            loader(complete(vec![]), Default::default(), loads.clone()),
        )
        .await
        .unwrap();
        // Mutation that reds this: skipping `verify_source_run` / the plan-identity comparison.
        assert_eq!(loads.load(Ordering::SeqCst), 0, "{id}");
        let terminal = h.terminal();
        assert_eq!(terminal["status"], "failed", "{id}: {terminal}");
    }

    // The matching plan restores with its recorded plan identity (never re-planned).
    let seen = Arc::new(Mutex::new(Seen::default()));
    let job = h.job(
        "plan-ok",
        json!({"kind": "fromPlan", "planJobId": "job-src", "steps": 4,
               "semanticSampling": {"topK": 50}, "decoder": "legacy",
               "sources": {"plan": {"jobId": "job-src", "runDir": "yue2/runs/yue2run_src",
                                     "identity": "cafe01", "planIdentity": "ab12"}}}),
    );
    h.run(
        &job,
        loader(complete(vec![]), seen.clone(), Default::default()),
    )
    .await
    .unwrap();
    assert_eq!(h.terminal()["status"], "completed");
    let seen = seen.lock().unwrap();
    let req = seen.request.as_ref().unwrap();
    let song = req.audio.as_ref().unwrap().song.clone().unwrap();
    let plan = song.plan.unwrap();
    assert_eq!(plan.dir, h.run_dir("yue2run_src"));
    assert_eq!(plan.identity.as_deref(), Some("ab12"));
    assert_eq!(
        (req.seed, req.guidance),
        (None, None),
        "a saved plan fixes them"
    );
    assert_eq!(req.steps, Some(4));
    assert_eq!(song.semantic_sampling.unwrap().top_k, Some(50));
}

/// A failed run persists with its provenance and keeps its checkpoints for a resume; a truncated
/// run is recorded as truncated on the result and the asset.
#[tokio::test]
async fn failed_and_truncated_runs_persist_with_their_provenance() {
    let h = Harness::new().await;
    let job = h.job(
        "fail-1",
        json!({"kind": "create", "style": "x", "lyrics": "[verse]\nla", "seed": 1}),
    );
    h.run(
        &job,
        loader(Behavior::Fail, Default::default(), Default::default()),
    )
    .await
    .unwrap();
    let terminal = h.terminal();
    assert_eq!(terminal["status"], "failed");
    let block = &terminal["result"]["yue2"];
    assert_eq!(block["status"], "failed");
    assert_eq!(block["run"]["dir"], "yue2/runs/yue2run_fail-1");
    assert_eq!(block["run"]["partial"], true);
    assert_eq!(block["effectiveSettings"]["seed"], 1);
    assert_eq!(block["effectiveSettings"]["tier"], "bf16");
    assert_eq!(block["usagePolicy"]["checkedAt"], "execution");
    assert!(block["error"]
        .as_str()
        .unwrap()
        .contains("semantic stage failed"));
    // Mutation that reds this: removing the working directory on a failure (only a cancel does).
    assert!(partial_dir(&h.run_dir("yue2run_fail-1"))
        .join("plan.json")
        .is_file());

    drop(h);
    let h = Harness::new().await;
    let job = h.job(
        "trunc-1",
        json!({"kind": "create", "style": "lofi", "lyrics": "[verse]\nla"}),
    );
    h.run(
        &job,
        loader(
            Behavior::Complete {
                steps: vec![],
                truncated: (true, true),
            },
            Default::default(),
            Default::default(),
        ),
    )
    .await
    .unwrap();
    let result = h.terminal()["result"].clone();
    assert_eq!(
        result["yue2"]["truncated"],
        json!({"abc": true, "semantic": true})
    );
    let codes: Vec<&str> = result["yue2"]["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["code"].as_str().unwrap())
        .collect();
    assert_eq!(codes, ["abc_truncated", "semantic_truncated"]);
    let fact = &result["assetWrites"][0];
    assert_eq!(fact["displayName"], "lofi (truncated)");
    assert_eq!(fact["extra"]["yue2"]["truncated"]["semantic"], true);
}

/// Tier and decoder availability are checked before load: an underived q8 tier, a legacy decoder
/// that is not installed, and a blocked transcription are refused with the reason.
#[tokio::test]
async fn missing_tiers_decoders_and_blocked_transcription_refuse_before_load() {
    let h = Harness::with_decoders(true, false).await;
    for (id, spec, needle) in [
        (
            "tier-q8",
            json!({"kind": "create", "lyrics": "[verse]\nla", "tier": "q8"}),
            "has not been derived",
        ),
        (
            "legacy",
            json!({"kind": "create", "lyrics": "[verse]\nla", "decoder": "legacy"}),
            "legacy decoder is not installed",
        ),
        (
            "transcribe",
            json!({"kind": "transcribe", "sourceAudioAssetId": "asset-1"}),
            "component_blocked",
        ),
    ] {
        let loads = Arc::new(AtomicUsize::new(0));
        let job = h.job(id, spec);
        h.run(
            &job,
            loader(complete(vec![]), Default::default(), loads.clone()),
        )
        .await
        .unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 0, "{id}");
        let terminal = h.terminal();
        assert_eq!(terminal["status"], "failed", "{id}");
        assert!(
            terminal["error"].as_str().unwrap().contains(needle),
            "{id}: {terminal}"
        );
    }
}

/// A derived tier that verifies against its pins loads as that tier (`quantize` asserted).
#[tokio::test]
async fn a_verified_derived_tier_loads_with_its_quantization() {
    let h = Harness::new().await;
    let row = h.entry["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["variant"] == "q4")
        .unwrap()
        .clone();
    // Re-pin the q4 row to bytes this test can produce, then stage them where the catalog reads.
    let bytes = b"q4 weights".to_vec();
    let mut entry = h.entry.clone();
    for row in entry["downloads"].as_array_mut().unwrap() {
        if row["variant"] == "q4" {
            row["localDerivation"]["weightsBytes"] = json!(bytes.len());
            row["localDerivation"]["weightsSha256"] = json!(sha256_hex(&bytes));
        }
    }
    let derivation = local_derivation(&row).unwrap();
    let dir = local_derivation_snapshot_dir(h.data.path(), "yue2", "q4", &derivation).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
    let mut job = h.job(
        "q4-1",
        json!({"kind": "create", "lyrics": "[verse]\nla", "tier": "q4"}),
    );
    job.payload.insert("modelManifestEntry".into(), entry);
    let seen = Arc::new(Mutex::new(Seen::default()));
    h.run(
        &job,
        loader(complete(vec![]), seen.clone(), Default::default()),
    )
    .await
    .unwrap();
    assert_eq!(h.terminal()["status"], "completed");
    let seen = seen.lock().unwrap();
    let load = seen.load.as_ref().unwrap();
    assert_eq!(load.quantize, Some(Quant::Q4));
    assert!(matches!(&load.weights, WeightsSource::Dir(d) if *d == dir));
}

/// The worker's copies of the engine's run-layout names are the linked engine's.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
#[test]
fn run_layout_names_match_the_linked_engine() {
    use crate::inference_runtime::candle_audio_yue2::run;
    assert_eq!(PARTIAL_SUFFIX, run::PARTIAL_SUFFIX);
    assert_eq!(RUN_LOCK_FILE, run::LOCK_FILE);
    let dir = Path::new("/p/yue2/runs/yue2run_x");
    assert_eq!(partial_dir(dir), run::partial_dir(dir));
}

/// The real-weight end-to-end run (sc-22999): a plan-only job, a short song and a restored-plan
/// synthesis through the SceneWorks worker's job path, on the CPU.
///
/// On macOS the registered provider always runs on Metal (`runtime-macos` builds the audio lane
/// with `audio-metal` and `candle_audio::default_device` has no CPU override). This lane is
/// CPU-only, so the loader injected here builds the SAME engine from the SAME verified snapshots on
/// `Device::Cpu` and adapts it through the provider's own request mapping (`map_request` /
/// `memory_options`) — everything else (eligibility, resolution, the run directory, progress,
/// cancel, publication) is the production job path.
///
/// Run: `SCENEWORKS_YUE2_E2E=1 cargo test -p sceneworks-worker --lib -- --ignored
/// yue2_jobs::tests::real_weights --nocapture` (weights in `SCENEWORKS_YUE2_HF_HUB`, default
/// `/Volumes/Models/huggingface/hub`). About 12 GB resident (F32 on the CPU).
#[cfg(target_os = "macos")]
mod real_weights {
    use super::*;
    use crate::inference_runtime::candle_audio_yue2 as y2;
    use runtime_macos::audio::candle_core::{DType, Device};

    struct CpuYue2 {
        descriptor: gen_core::ModelDescriptor,
        engine: y2::Yue2Engine,
    }

    struct Bridge<'a> {
        on_progress: &'a mut dyn FnMut(Progress),
        abc_total: u32,
        semantic_total: u32,
        tokens: u32,
    }

    impl y2::EngineObserver for Bridge<'_> {
        fn on_stage(&mut self, stage: y2::Stage, event: y2::StageEvent) {
            match (stage, event) {
                (y2::Stage::Plan | y2::Stage::Semantic, y2::StageEvent::Started) => self.tokens = 0,
                (y2::Stage::Decode, y2::StageEvent::Started) => {
                    (self.on_progress)(Progress::Decoding)
                }
                _ => {}
            }
        }
        fn on_token(&mut self, stage: y2::Stage, _token: u32) {
            let total = match stage {
                y2::Stage::Plan => self.abc_total,
                y2::Stage::Semantic => self.semantic_total,
                _ => return,
            };
            self.tokens += 1;
            (self.on_progress)(Progress::Step {
                current: self.tokens.min(total),
                total,
            });
        }
        fn on_synthesis_progress(&mut self, completed: usize, total: usize) {
            (self.on_progress)(Progress::Step {
                current: completed as u32,
                total: total as u32,
            });
        }
    }

    fn record(result: &Value, dir: &Path) -> gen_core::ArtifactRecord {
        gen_core::ArtifactRecord {
            dir: dir.to_path_buf(),
            kind: result["kind"].as_str().unwrap().to_owned(),
            identity: result["identity"].as_str().unwrap().to_owned(),
        }
    }

    fn warnings(result: &Value) -> Vec<gen_core::GenerationWarning> {
        ["abc", "semantic"]
            .into_iter()
            .filter(|key| result.pointer(&format!("/truncated/{key}")) == Some(&json!(true)))
            .map(|key| gen_core::GenerationWarning {
                code: format!("{key}_truncated"),
                message: format!("the {key} phase hit its max_tokens budget"),
            })
            .collect()
    }

    impl Generator for CpuYue2 {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            Err(gen_core::Error::Msg("generate_with_report only".into()))
        }
        fn generate_with_report(
            &self,
            req: &GenerationRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationReport> {
            let engine = &self.engine;
            let mut mapped = y2::provider::map_request(req, engine.generation_config())?;
            mapped.settings.options = y2::provider::memory_options(
                req.memory.as_ref(),
                engine.options(),
                engine.attention_bounds(),
            )?;
            let options = engine.options_for(&mapped.settings);
            let cancel = req.cancel.clone();
            let cancelled = move || cancel.is_cancelled();
            let generation = mapped.settings.generation.clone();
            let mut bridge = Bridge {
                on_progress,
                abc_total: generation.abc().max_tokens() as u32,
                semantic_total: generation.semantic().max_tokens() as u32,
                tokens: 0,
            };
            let mut hooks = y2::EngineHooks {
                cancelled: &cancelled,
                observer: &mut bridge,
            };
            let output = mapped
                .output
                .clone()
                .expect("the job always asks for artifacts");
            let audio = |samples: Vec<f32>| {
                GenerationOutput::Audio(gen_core::AudioTrack {
                    samples,
                    sample_rate: 48_000,
                    channels: 2,
                    stems: Vec::new(),
                })
            };
            let (samples, result, dir) = match mapped.job {
                y2::provider::Job::DecodeCached(source) => {
                    let outcome = engine
                        .decode_cached_with(
                            &source,
                            mapped.settings.decoder,
                            &options.decode,
                            Some(&output),
                            &mut hooks,
                        )
                        .map_err(gen_core::Error::from)?;
                    (Some(outcome.samples), outcome.result, outcome.dir)
                }
                y2::provider::Job::Generate(request) if mapped.plan_only => {
                    let (_, _, dir) = engine
                        .plan_to(&request, &generation, &output, &mut hooks)
                        .map_err(gen_core::Error::from)?;
                    let result = y2::run::verify_run(&dir, None).map_err(gen_core::Error::from)?;
                    (None, result, dir)
                }
                job => {
                    let input = match job {
                        y2::provider::Job::Generate(request) => {
                            y2::run::SongInput::Request(request)
                        }
                        y2::provider::Job::FromPlan { dir, identity } => {
                            let plan = y2::SymbolicPlan::restore(&dir, engine.tokenizer())
                                .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
                            assert_eq!(Some(plan.identity().to_string()), identity);
                            y2::run::SongInput::Plan(plan)
                        }
                        y2::provider::Job::DecodeCached(_) => unreachable!(),
                    };
                    let outcome = engine
                        .generate_to(&input, &mapped.settings, &output, &mut hooks)
                        .map_err(gen_core::Error::from)?;
                    (Some(outcome.samples), outcome.result, outcome.dir)
                }
            };
            Ok(GenerationReport {
                output: samples.map(audio),
                artifacts: Some(record(&result, &dir)),
                warnings: warnings(&result),
            })
        }
    }

    fn cpu_loader() -> Loader {
        Box::new(|id: &str, spec: &LoadSpec| {
            assert_eq!(id, "yue2");
            let WeightsSource::Dir(weights) = &spec.weights else {
                panic!("the job stages a snapshot directory")
            };
            let mut dirs = y2::SnapshotDirs::new()
                .with(y2::ComponentId::Lm.component().repo.id, weights.clone());
            for (component, id) in [
                ("vae", y2::ComponentId::VaeStandard),
                ("vae_legacy", y2::ComponentId::VaeLegacy),
            ] {
                if let Some(WeightsSource::Dir(dir)) = spec.components.get(component) {
                    dirs = dirs.with(id.component().repo.id, dir.clone());
                }
            }
            let engine = y2::Yue2Engine::load_with_precision(
                &dirs,
                DType::F32,
                &Device::Cpu,
                y2::engine::ModelPrecision::default(),
                y2::GenerationConfig::default(),
                y2::provider::engine_options(spec),
            )?;
            Ok(Box::new(CpuYue2 {
                descriptor: y2::descriptor(),
                engine,
            }) as Box<dyn Generator>)
        })
    }

    fn hub() -> String {
        assert!(
            std::env::var("SCENEWORKS_YUE2_E2E").is_ok_and(|v| v == "1"),
            "set SCENEWORKS_YUE2_E2E=1 to run the real-weight YuE2 end-to-end test"
        );
        std::env::var("SCENEWORKS_YUE2_HF_HUB")
            .unwrap_or_else(|_| "/Volumes/Models/huggingface/hub".to_owned())
    }

    const LYRICS: &str = "[verse]\nMorning light on the harbor\nBoats are coming home\n";
    const STYLE: &str = "folk, acoustic guitar, female vocal, warm";

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "real YuE2 weights on the CPU; SCENEWORKS_YUE2_E2E=1"]
    async fn real_weights_plan_song_and_restored_plan_on_cpu_through_the_worker() {
        let hub = hub();
        let h = Harness::over_hub(&hub).await;
        let started = Instant::now();

        // 1. Plan only.
        let plan = h.job(
            "e2e-plan",
            json!({"kind": "plan", "style": STYLE, "lyrics": LYRICS, "seed": 11,
                   "scoreSampling": {"minTokens": 8, "maxTokens": 96}}),
        );
        h.run(&plan, cpu_loader()).await.expect("the plan job runs");
        let terminal = h.terminal();
        assert_eq!(terminal["status"], "completed", "{terminal}");
        let block = terminal["result"]["yue2"].clone();
        assert_eq!(block["run"]["kind"], "plan");
        let plan_dir = h.run_dir("yue2run_e2e-plan");
        crate::inference_runtime::verify_yue2_run(&plan_dir, block["run"]["identity"].as_str())
            .expect("the published plan verifies");
        eprintln!(
            "plan: {:?} truncated={}",
            started.elapsed(),
            block["truncated"]
        );

        // 2. A short song.
        let song_started = Instant::now();
        let song = h.job(
            "e2e-song",
            json!({"kind": "create", "style": STYLE, "lyrics": LYRICS, "seed": 12, "steps": 2,
                   "scoreSampling": {"minTokens": 8, "maxTokens": 96},
                   "semanticSampling": {"minTokens": 20, "maxTokens": 150}}),
        );
        h.run(&song, cpu_loader()).await.expect("the song job runs");
        let terminal = h.terminal();
        assert_eq!(terminal["status"], "completed", "{terminal}");
        let fact = terminal["result"]["assetWrites"][0].clone();
        assert!(fact["duration"].as_f64().unwrap() > 0.0, "{fact}");
        assert_eq!(fact["sampleRate"], 48_000);
        assert!(h
            .project_path
            .join(fact["mediaPath"].as_str().unwrap())
            .is_file());
        let steps = h
            .progress()
            .iter()
            .filter(|p| p["status"] == "running")
            .map(|p| p["message"].as_str().unwrap_or("").to_owned())
            .collect::<Vec<_>>();
        assert!(
            steps
                .iter()
                .any(|m| m.starts_with("Generating semantic tokens")),
            "{steps:?}"
        );
        eprintln!(
            "song: {:?} duration={}s truncated={} warnings={}",
            song_started.elapsed(),
            fact["duration"],
            terminal["result"]["yue2"]["truncated"],
            terminal["result"]["yue2"]["warnings"]
        );

        // 3. Synthesis from the saved plan of step 1, verified against its recorded identity.
        let restored_started = Instant::now();
        let from_plan = h.job(
            "e2e-from-plan",
            json!({"kind": "fromPlan", "planJobId": "job-e2e-plan", "steps": 2,
                   "semanticSampling": {"minTokens": 20, "maxTokens": 150},
                   "sources": {"plan": {"jobId": "job-e2e-plan",
                                         "runDir": "yue2/runs/yue2run_e2e-plan",
                                         "identity": block["run"]["identity"],
                                         "planIdentity": block["run"]["planIdentity"]}}}),
        );
        h.run(&from_plan, cpu_loader())
            .await
            .expect("the restored-plan job runs");
        let terminal = h.terminal();
        assert_eq!(terminal["status"], "completed", "{terminal}");
        eprintln!(
            "fromPlan: {:?}; total {:?}",
            restored_started.elapsed(),
            started.elapsed()
        );
    }

    /// Resuming a COMPLETED run reuses it whatever memory controls the resuming request carries —
    /// memory controls change no sample and must not enter the run identity. (Fails at inference
    /// accd49ea, where they leaked into the identity; passes once the inference fix is pinned.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "real YuE2 weights on the CPU; SCENEWORKS_YUE2_E2E=1"]
    async fn real_weights_resume_of_a_completed_run_ignores_memory_controls() {
        let hub = hub();
        let h = Harness::over_hub(&hub).await;
        let spec = json!({"kind": "create", "style": STYLE, "lyrics": LYRICS, "seed": 21,
                          "steps": 2, "runId": "yue2run_e2e-resume",
                          "scoreSampling": {"minTokens": 8, "maxTokens": 64},
                          "semanticSampling": {"minTokens": 20, "maxTokens": 100}});
        h.run(&h.job("e2e-resume-1", spec.clone()), cpu_loader())
            .await
            .unwrap();
        let first = h.terminal();
        assert_eq!(first["status"], "completed", "{first}");
        let mut resumed = spec;
        resumed["memory"] = json!({"tileVaeDecode": true, "decodeTileEdge": 32});
        let started = Instant::now();
        h.run(&h.job("e2e-resume-2", resumed), cpu_loader())
            .await
            .unwrap();
        let second = h.terminal();
        assert_eq!(second["status"], "completed", "{second}");
        assert_eq!(
            second["result"]["yue2"]["run"]["identity"], first["result"]["yue2"]["run"]["identity"],
            "the completed run was reused, not recomputed"
        );
        eprintln!("resume: {:?}", started.elapsed());
    }
}
