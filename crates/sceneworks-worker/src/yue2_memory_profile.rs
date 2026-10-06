//! One fresh-process YuE2 memory-profile capture (sc-23001) — the native half of
//! `scripts/yue2-memory-profile.mjs`, which the terminal story (sc-23002) runs ONCE over the cases in
//! `config/yue2-memory-profile-plan.json`. Nothing here runs in an ordinary test run.
//!
//! The Node controller owns the case list, repository and closure-digest identity, the memory
//! sampler (Darwin: the `memory-calibration-watchdog.py` footprint guard; CUDA: `nvidia-smi`) and
//! the record. This module owns the reviewable native work, through the job path's OWN functions
//! (sc-23002): a case is parsed into the `create` job the API would queue ([`case_spec`]), whose
//! weights are resolved by `yue2_jobs::resolve_load` (the pinned bf16 snapshot, or the locally derived
//! q8 / q4 tier verified against its pins), whose engine request is `yue2_jobs::build_request`, and
//! whose admission is the job's gate ([`crate::yue2_admission::check`] with `yue2_jobs::admission_pins`,
//! against this host's live budget); the registered `yue2` provider is loaded through
//! [`crate::inference_runtime::load_audio`]. It writes, into the case's output directory:
//!
//! * `admission.json` — the admission outcome: refused (with the gate's message), or the chosen
//!   controls, the per-stage estimate and its evidence class;
//! * `stages.jsonl` — `{"stage", "at"}` marks (wall-clock seconds, the sampler's clock) as the
//!   render moves through load → plan → semantic → acoustic → decode, driven by the admission lease
//!   from the engine's own progress, so the controller attributes each memory sample to a stage;
//! * `outcome.json` — completed, with the audio's length and RMS, the run's truncation flags, the
//!   per-stage wall times (from the stage marks, [`stage_seconds`], including time waiting for each
//!   boundary sample), the engine's own timings and the
//!   run / plan / decoder / latent identities read back from the published run's `result.json`
//!   ([`outcome_json`]) — so a record states which exact latent and decoder it measured.
//!
//! ```text
//! SCENEWORKS_ENABLE_YUE2_MEMORY_PROFILE=1 SCENEWORKS_YUE2_PROFILE_CASE=<case.json> \
//!   SCENEWORKS_YUE2_PROFILE_OUT=<dir> SCENEWORKS_DATA_DIR=<app data dir> \
//!   cargo test --release -p sceneworks-worker --lib yue2_memory_profile::capture_case -- \
//!   --ignored --exact --nocapture --test-threads 1
//! ```

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use gen_core::{CancelFlag, GenerationOutput, GenerationRequest, Progress};
use sceneworks_core::yue2_score::jobs::{self as contract, Yue2JobSpec};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::yue2_admission::{Yue2LoadFacts, Yue2Stage, Yue2Tier};
use crate::yue2_jobs::{admission_pins, build_request, resolve_load, Inputs};
use crate::Settings;

const ENABLE_ENV: &str = "SCENEWORKS_ENABLE_YUE2_MEMORY_PROFILE";
const CASE_ENV: &str = "SCENEWORKS_YUE2_PROFILE_CASE";
const OUT_ENV: &str = "SCENEWORKS_YUE2_PROFILE_OUT";
const BOUNDARY_ENV: &str = "SCENEWORKS_YUE2_PROFILE_BOUNDARY";
const SAMPLER_ENV: &str = "SCENEWORKS_YUE2_PROFILE_SAMPLER";

/// One AR phase's sampling overrides, as the plan spells them.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CaseSampling {
    #[serde(default)]
    min_tokens: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

/// The request a case renders — a subset of the YuE2 job contract, enough to shape memory.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CaseRequest {
    planning: String,
    style: String,
    lyrics: String,
    seed: u64,
    #[serde(default)]
    cfg_scale: Option<f64>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    score_sampling: Option<CaseSampling>,
    #[serde(default)]
    semantic_sampling: Option<CaseSampling>,
}

/// A case file the controller writes (`scripts/yue2-memory-profile.mjs` `caseFile`).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ProfileCase {
    id: String,
    tier: String,
    decoder: String,
    #[serde(default)]
    ar_mode: Option<contract::ArMode>,
    #[serde(default)]
    compute_policy: Option<contract::ComputePolicy>,
    request: CaseRequest,
}

fn sampling_json(s: &CaseSampling) -> Value {
    json!({ "minTokens": s.min_tokens, "maxTokens": s.max_tokens })
}

/// The YuE2 job a case measures: the `create` job the API queues for the same request, parsed from
/// the same `POST /api/v1/projects/:id/yue2/jobs` body and held to the same `validate_request`. The
/// capture then resolves, admits and renders it through the job's own `resolve_load`,
/// `admission_pins` and `build_request` (`crate::yue2_jobs`), so the campaign measures exactly what a
/// production job runs — there is no second copy of the load or request construction to drift.
pub(crate) fn case_spec(case: &ProfileCase) -> Result<Yue2JobSpec, String> {
    let request = &case.request;
    let body = json!({
        "kind": "create",
        "planning": request.planning,
        "style": request.style,
        "lyrics": request.lyrics,
        "seed": request.seed,
        "cfgScale": request.cfg_scale,
        "steps": request.steps,
        "scoreSampling": request.score_sampling.as_ref().map(sampling_json),
        "semanticSampling": request.semantic_sampling.as_ref().map(sampling_json),
        "decoder": case.decoder,
        "tier": case.tier,
        "arMode": case.ar_mode,
        "computePolicy": case.compute_policy,
    });
    let spec: Yue2JobSpec =
        serde_json::from_value(body).map_err(|error| format!("{}: {error}", case.id))?;
    contract::validate_request(&spec).map_err(|error| format!("{}: {error}", case.id))?;
    Ok(spec)
}

/// The request a case renders into `run_dir`: the job path's `build_request` for [`case_spec`].
pub(crate) fn case_request(
    case: &ProfileCase,
    run_dir: &Path,
) -> Result<GenerationRequest, String> {
    Ok(build_request(
        &case_spec(case)?,
        &Inputs::default(),
        run_dir,
        CancelFlag::new(),
    ))
}

/// The stage key a record carries for a lease stage (the acoustic prefill and solve alternate chunk
/// by chunk, so they are one measured stage).
pub(crate) fn stage_key(stage: Yue2Stage) -> &'static str {
    match stage {
        Yue2Stage::AcousticPrefill | Yue2Stage::AcousticSolve => "acoustic",
        other => other.key(),
    }
}

/// The per-stage estimate a record compares its measurement with: each measured stage's device and
/// host bytes (the acoustic stage is the larger of its two phases on each pool).
pub(crate) fn estimate_json(estimate: &crate::yue2_admission::Yue2Estimate) -> Value {
    let mut pools: std::collections::BTreeMap<&str, (u64, u64)> = Default::default();
    for stage in &estimate.stages {
        let entry = pools.entry(stage_key(stage.stage)).or_default();
        entry.0 = entry.0.max(stage.device_bytes());
        entry.1 = entry.1.max(stage.host_bytes());
    }
    let stages: serde_json::Map<String, Value> = pools
        .into_iter()
        .map(|(key, (device, host))| {
            (
                key.to_owned(),
                json!({ "deviceBytes": device, "hostBytes": host, "totalBytes": device + host }),
            )
        })
        .collect();
    json!({
        "tier": estimate.tier.key(),
        "backend": estimate.backend.key(),
        "evidence": crate::yue2_admission::Yue2Evidence::of(estimate.backend).key(),
        "controls": {
            "stageResidency": estimate.controls.offload_ar,
            "attentionChunkSize": estimate.controls.attention_elements,
            "decodeTileEdge": estimate.controls.decode_core_frames,
        },
        "weights": {
            "deviceBytes": estimate.weights.device_bytes,
            "hostBytes": estimate.weights.host_bytes,
        },
        "stages": stages,
    })
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn write_json(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec_pretty(value).expect("json")).expect("write json");
}

struct StageMarks {
    file: std::fs::File,
    boundary: StageBoundary,
    /// Every mark written, in order: `(stage, at)`.
    marks: Vec<(&'static str, f64)>,
}

impl StageMarks {
    fn open(path: &Path, boundary_dir: &Path, sampler: &'static str) -> Self {
        Self {
            file: std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("open stage marks"),
            boundary: StageBoundary::new(boundary_dir, sampler),
            marks: Vec::new(),
        }
    }

    fn mark(&mut self, stage: &'static str) {
        if self.marks.last().map(|(last, _)| *last) == Some(stage) {
            return;
        }
        if let Some((previous, _)) = self.marks.last() {
            self.boundary.wait_for_sample(previous);
        }
        let at = now_secs();
        self.marks.push((stage, at));
        let line = json!({ "stage": stage, "at": at });
        writeln!(self.file, "{line}").expect("write stage mark");
        self.file.flush().expect("flush stage mark");
        self.file.sync_data().expect("sync stage mark");
    }
}

/// The controller alone can acknowledge a reading from its external sampler. A boundary holds
/// the old stage (and, for decode, the loaded generator) until that reading is committed.
struct StageBoundary {
    dir: PathBuf,
    sequence: u64,
    sampler: &'static str,
    last_serial: u64,
}

impl StageBoundary {
    fn new(dir: &Path, sampler: &'static str) -> Self {
        Self {
            dir: dir.to_owned(),
            sequence: 0,
            sampler,
            last_serial: 0,
        }
    }

    fn wait_for_sample(&mut self, stage: &str) {
        self.wait_for_sample_until(stage, Duration::from_secs(60));
    }

    fn wait_for_sample_until(&mut self, stage: &str, timeout: Duration) {
        self.sequence += 1;
        let request = self.dir.join("request.json");
        let ack = self.dir.join("ack.json");
        match std::fs::remove_file(&ack) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => panic!("remove previous stage ack: {error}"),
        }
        let requested_at = now_secs();
        let value = json!({"sequence": self.sequence, "stage": stage, "requestedAt": requested_at,
            "processId": std::process::id()});
        let temp = self.dir.join("request.tmp");
        let mut file = std::fs::File::create(&temp).expect("create stage request");
        writeln!(file, "{value}").expect("write stage request");
        file.sync_all().expect("sync stage request");
        drop(file);
        match std::fs::remove_file(&request) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => panic!("remove previous stage request: {error}"),
        }
        std::fs::rename(temp, request).expect("publish stage request");
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(bytes) = std::fs::read(&ack) {
                let value: Value = serde_json::from_slice(&bytes).expect("invalid stage ack");
                assert_eq!(
                    value["sequence"].as_u64(),
                    Some(self.sequence),
                    "stale stage ack"
                );
                assert_eq!(value["stage"].as_str(), Some(stage), "mismatched stage ack");
                assert!(
                    value["sampleAt"]
                        .as_f64()
                        .is_some_and(|at| at >= requested_at),
                    "stage ack predates request"
                );
                assert!(
                    value["sampleBytes"].as_u64().is_some_and(|bytes| bytes > 0),
                    "stage ack has no external reading"
                );
                let serial = value["sampleSerial"]
                    .as_u64()
                    .expect("stage ack has no external serial");
                assert!(serial > self.last_serial, "stale stage sample serial");
                assert_eq!(
                    value["sampler"].as_str(),
                    Some(self.sampler),
                    "wrong stage sampler"
                );
                self.last_serial = serial;
                return;
            }
            assert!(
                Instant::now() < deadline,
                "stage {stage} sample acknowledgment timed out"
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

/// The wall time each stage held, including the wait for its external boundary sample, in seconds:
/// from its mark to the next mark. A stage the render
/// entered more than once (it cannot today — the lease only moves forward) sums its spans. The last
/// mark (`done`) closes the final stage and holds no time of its own.
pub(crate) fn stage_seconds(marks: &[(&'static str, f64)]) -> serde_json::Map<String, Value> {
    let mut seconds: std::collections::BTreeMap<&str, f64> = Default::default();
    for pair in marks.windows(2) {
        let ((stage, from), (_, to)) = (pair[0], pair[1]);
        *seconds.entry(stage).or_default() += (to - from).max(0.0);
    }
    seconds
        .into_iter()
        .map(|(stage, secs)| (stage.to_owned(), json!(secs)))
        .collect()
}

/// `outcome.json` for a completed render: the audio's length and RMS, the published run's truncation
/// flags and identities (`result.json`: run, plan, decoder, latent) and its engine timings, and the
/// per-stage wall times. Every run field is required — a run record that lacks one is an error, never
/// a default, so a record can never claim an identity or a truncation state the run did not state.
struct OutcomeModes<'a> {
    ar_mode: contract::ArMode,
    compute_policy: Option<contract::ComputePolicy>,
    backend: crate::yue2_admission::Yue2Backend,
    engine_config: Option<&'a Value>,
}

fn outcome_json(
    case_id: &str,
    audio_seconds: f64,
    rms: f32,
    run_result: &Value,
    modes: OutcomeModes<'_>,
    stage_seconds: serde_json::Map<String, Value>,
) -> Result<Value, String> {
    let field = |key: &str| {
        run_result
            .get(key)
            .filter(|value| !value.is_null())
            .cloned()
            .ok_or_else(|| format!("the run's result.json has no `{key}`"))
    };
    let truncated = field("truncated")?;
    for phase in ["abc", "semantic"] {
        if !truncated.get(phase).is_some_and(Value::is_boolean) {
            return Err(format!(
                "the run's result.json states no boolean `truncated.{phase}`"
            ));
        }
    }
    let latent = field("latent")?;
    if !latent
        .get("sha256")
        .and_then(Value::as_str)
        .is_some_and(|sha| sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("the run's latent identity has no sha256".into());
    }
    let mut outcome = json!({
        "caseId": case_id,
        "outcome": "completed",
        "audioSeconds": audio_seconds,
        "rms": rms,
        "truncated": { "abc": truncated["abc"], "semantic": truncated["semantic"] },
        "stageSeconds": stage_seconds,
        "engineTiming": field("timing")?,
        "runIdentity": field("identity")?,
        "planIdentity": field("plan_identity")?,
        "decoder": field("decoder")?,
        "latent": latent,
    });
    if modes.ar_mode == contract::ArMode::ExperimentalFp8 {
        let quantization = modes
            .engine_config
            .and_then(|config| config.get("quantization"))
            .and_then(Value::as_str)
            .ok_or_else(|| "the run's config.json has no effective quantization".to_owned())?;
        if quantization != "fp8" {
            return Err(format!(
                "the run's effective AR quantization is {quantization}, not requested fp8"
            ));
        }
        outcome["engineQuantization"] = json!(quantization);
    }
    if let Some(policy) = modes.compute_policy {
        let expected = match policy {
            contract::ComputePolicy::Auto => ("auto", "bfloat16", "float32"),
            contract::ComputePolicy::Bf16 => ("bf16", "bfloat16", "bfloat16"),
            contract::ComputePolicy::Fp32 => ("fp32", "float32", "float32"),
        };
        let config = modes
            .engine_config
            .ok_or("the run has no effective config.json")?;
        for (key, wanted) in [
            ("compute_policy", expected.0),
            ("model_dtype", expected.1),
            ("vae_dtype", expected.2),
        ] {
            if config.get(key).and_then(Value::as_str) != Some(wanted) {
                return Err(format!(
                    "the run's effective {key} is not requested {wanted}"
                ));
            }
        }
        outcome["engineComputePolicy"] = json!(expected.0);
        outcome["engineModelDtype"] = json!(expected.1);
        outcome["engineVaeDtype"] = json!(expected.2);
        // The backend is the admission gate's selected device, never a case-file claim. Only
        // CUDA's strict BF16 VAE uses fixed-order convolution; all other paths must omit its policy.
        let math_policy = config
            .get("vae_cuda_bf16_math_policy")
            .and_then(Value::as_str);
        if modes.backend == crate::yue2_admission::Yue2Backend::Cuda
            && policy == contract::ComputePolicy::Bf16
        {
            const EXPECTED: &str = "fixed_order_bf16_convolution_v1";
            if math_policy != Some(EXPECTED) {
                return Err(format!(
                    "the run's effective vae_cuda_bf16_math_policy is not {EXPECTED}"
                ));
            }
            outcome["engineVaeCudaBf16MathPolicy"] = json!(EXPECTED);
        } else if config.get("vae_cuda_bf16_math_policy").is_some() {
            return Err(
                "the run has a CUDA BF16 VAE math policy on another backend or compute policy"
                    .into(),
            );
        }
    }
    Ok(outcome)
}

fn builtin_entry() -> Value {
    crate::yue2_admission::tests::builtin_yue2_entry()
}

/// The single capture entrypoint. Refusals are results (recorded, not failures); a load or render
/// error fails the test so the controller records the case as failed.
#[test]
#[ignore = "real weights + GPU: run only by scripts/yue2-memory-profile.mjs capture (sc-23002)"]
fn capture_case() {
    assert_eq!(
        std::env::var(ENABLE_ENV).as_deref(),
        Ok("1"),
        "set {ENABLE_ENV}=1 to run a YuE2 memory-profile capture"
    );
    let case_file = PathBuf::from(std::env::var(CASE_ENV).expect(CASE_ENV));
    let out = PathBuf::from(std::env::var(OUT_ENV).expect(OUT_ENV));
    std::fs::create_dir_all(&out).expect("output dir");
    let case: ProfileCase =
        serde_json::from_slice(&std::fs::read(&case_file).expect("read case")).expect("case json");
    let settings = Settings::from_env();
    let entry = builtin_entry();
    let run_dir = out.join("run");
    // A capture measures one fresh render: a run left by an earlier attempt would be resumed (the
    // job path renders with `resume: true`) and measure a partial render.
    for leftover in [run_dir.clone(), crate::yue2_jobs::partial_dir(&run_dir)] {
        assert!(
            !leftover.exists(),
            "{leftover:?} already exists; capture into a fresh output directory"
        );
    }
    let process_file = out.join("profile-process.json");
    let mut process_out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&process_file)
        .expect("refuse an old profile process identity");
    writeln!(
        process_out,
        "{}",
        json!({ "processId": std::process::id() })
    )
    .expect("write profile process identity");
    process_out
        .sync_all()
        .expect("sync profile process identity");
    drop(process_out);
    let spec = case_spec(&case).unwrap_or_else(|why| panic!("{why}"));
    let load = resolve_load(&settings, &entry, &spec).unwrap_or_else(|why| panic!("{why}"));
    let mut request = build_request(&spec, &Inputs::default(), &run_dir, CancelFlag::new());
    let tier = Yue2Tier::from_key(load.tier.as_str()).expect("a YuE2 tier");
    let pins = admission_pins(&spec).unwrap_or_else(|why| panic!("{why}"));

    crate::yue2_admission::probe_hardware_in_this_test();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    // The normal worker discovers and caches its selected GPU before claiming a job. This
    // standalone capture enters admission directly, so bind a fresh reading for this GPU instead.
    // An unreadable capability remains None and the FP8 gate still refuses it.
    let _selected_cap = runtime.block_on(crate::yue2_admission::capture_selected_compute_cap(
        &settings.gpu_id,
        |gpu_id| async move { crate::gpu::nvidia_compute_cap(&gpu_id).await },
    ));
    let admitted = runtime.block_on(crate::yue2_admission::check(
        contract::MODEL_ID,
        &entry,
        &request,
        Yue2LoadFacts::of(tier, &load.spec),
        Some(pins),
        &settings.gpu_id,
    ));
    let admitted = match admitted {
        Ok(admitted) => admitted,
        Err(error) => {
            write_json(
                &out.join("admission.json"),
                &json!({ "caseId": case.id, "outcome": "refused", "message": error.to_string() }),
            );
            return;
        }
    };
    write_json(
        &out.join("admission.json"),
        &json!({
            "caseId": case.id,
            "outcome": "admitted",
            "estimate": estimate_json(admitted.lease.estimate()),
        }),
    );
    request.memory = Some(admitted.memory);
    let backend = admitted.lease.estimate().backend;
    let mut lease = admitted.lease;
    let boundary = PathBuf::from(std::env::var(BOUNDARY_ENV).expect(BOUNDARY_ENV));
    assert_eq!(
        boundary,
        out.join("boundary"),
        "stage boundary path must belong to this case"
    );
    let sampler = match std::env::var(SAMPLER_ENV).as_deref() {
        Ok("metal-watchdog") => "metal-watchdog",
        Ok("cuda-nvidia-smi") => "cuda-nvidia-smi",
        _ => panic!("{SAMPLER_ENV} must name the external sampler"),
    };
    let mut marks = StageMarks::open(&out.join("stages.jsonl"), &boundary, sampler);
    marks.mark("load");
    let generator =
        crate::inference_runtime::load_audio(contract::MODEL_ID, &load.spec).expect("load yue2");
    let report = {
        let mut on_progress = |progress: Progress| {
            lease.observe(&progress);
            marks.mark(stage_key(lease.stage()));
        };
        generator
            .generate_with_report(&request, &mut on_progress)
            .expect("render")
    };
    // Observe the final decode stage while the generator and lease still hold their residency.
    marks.mark("done");
    // The generator's memory goes first, then the lease that held it (the job's drop order).
    drop(generator);
    drop(lease);
    let published = report
        .artifacts
        .as_ref()
        .expect("the render published no run record");
    let run_result: Value = serde_json::from_slice(
        &std::fs::read(published.dir.join("result.json")).expect("read the run's result.json"),
    )
    .expect("the run's result.json is JSON");
    let engine_config = if case.ar_mode == Some(contract::ArMode::ExperimentalFp8)
        || case.compute_policy.is_some()
    {
        Some(
            serde_json::from_slice::<Value>(
                &std::fs::read(published.dir.join("config.json"))
                    .expect("read the run's effective config.json"),
            )
            .expect("the run's config.json is JSON"),
        )
    } else {
        None
    };
    let (seconds, rms) = match report.output {
        Some(GenerationOutput::Audio(track)) => {
            let n = track.samples.len().max(1);
            let rms = (track.samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
            (
                track.samples.len() as f64
                    / (f64::from(track.sample_rate) * f64::from(track.channels.max(1))),
                rms,
            )
        }
        _ => panic!("the render returned no audio"),
    };
    assert!(rms > 1e-3, "the render is silent (rms {rms})");
    let outcome = outcome_json(
        &case.id,
        seconds,
        rms,
        &run_result,
        OutcomeModes {
            ar_mode: case.ar_mode.unwrap_or_default(),
            compute_policy: case.compute_policy,
            backend,
            engine_config: engine_config.as_ref(),
        },
        stage_seconds(&marks.marks),
    )
    .unwrap_or_else(|why| panic!("{why}"));
    write_json(&out.join("outcome.json"), &outcome);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{SongDecoder, SongPlanning};

    fn wait_request(dir: &Path) -> Value {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Ok(body) = std::fs::read(dir.join("request.json")) {
                return serde_json::from_slice(&body).unwrap();
            }
            assert!(
                Instant::now() < deadline,
                "native stage request did not appear"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn test_ack(dir: &Path, request: &Value, sequence: u64) {
        let ack = json!({
            "sequence": sequence, "stage": request["stage"],
            "sampleAt": request["requestedAt"].as_f64().unwrap() + 1.0,
            "sampleBytes": 42, "sampleSerial": 2, "sampler": "metal-watchdog",
        });
        let temp = dir.join("ack.tmp");
        std::fs::write(&temp, serde_json::to_vec(&ack).unwrap()).unwrap();
        std::fs::rename(temp, dir.join("ack.json")).unwrap();
    }

    #[test]
    fn a_stage_cannot_advance_until_the_external_sample_is_acknowledged() {
        let fixture = tempfile::tempdir().unwrap();
        let dir = fixture.path().to_owned();
        let marks_path = dir.join("stages.jsonl");
        let mut marks = StageMarks::open(&marks_path, &dir, "metal-watchdog");
        marks.mark("load");
        let worker = std::thread::spawn(move || {
            marks.mark("plan");
            marks
        });
        let request = wait_request(&dir);
        assert_eq!(request["stage"], "load");
        assert_eq!(
            request["processId"].as_u64(),
            Some(u64::from(std::process::id()))
        );
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(
            std::fs::read_to_string(&marks_path)
                .unwrap()
                .lines()
                .count(),
            1
        );
        test_ack(&dir, &request, 1);
        let marks = worker.join().unwrap();
        assert_eq!(
            marks.marks.iter().map(|mark| mark.0).collect::<Vec<_>>(),
            ["load", "plan"]
        );
    }

    #[test]
    fn missing_or_stale_stage_ack_fails_closed() {
        let fixture = tempfile::tempdir().unwrap();
        let dir = fixture.path().to_owned();
        let mut boundary = StageBoundary::new(&dir, "metal-watchdog");
        let timed_out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            boundary.wait_for_sample_until("load", Duration::from_millis(40));
        }));
        assert!(timed_out.is_err());
        let request = wait_request(&dir);
        let writer_dir = dir.clone();
        let writer = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let next = loop {
                let value = wait_request(&writer_dir);
                if value["sequence"] == 2 {
                    break value;
                }
                assert!(Instant::now() < deadline, "second request did not appear");
                std::thread::sleep(Duration::from_millis(5));
            };
            test_ack(&writer_dir, &next, 1);
        });
        assert_eq!(request["sequence"], 1);
        let stale = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            boundary.wait_for_sample_until("load", Duration::from_secs(2));
        }));
        writer.join().unwrap();
        assert!(stale.is_err());
    }

    fn case(json: Value) -> Result<ProfileCase, serde_json::Error> {
        serde_json::from_value(json)
    }

    /// A case as the harness writes it (`caseFile` in scripts/yue2-memory-profile.mjs), setting
    /// every field a case can carry, and the Song Lab / API job body for the same song.
    fn full_case() -> ProfileCase {
        case(json!({
            "id": "yue2:q4:metal:long-context",
            "tier": "q4",
            "decoder": "legacy",
            "request": {
                "planning": "melody", "style": "rock", "lyrics": "[verse]\nla", "seed": 7,
                "cfgScale": 1.01, "steps": 32,
                "scoreSampling": { "maxTokens": 2048 },
                "semanticSampling": { "minTokens": 16000, "maxTokens": 16000 }
            }
        }))
        .expect("parses")
    }

    fn full_job_body() -> Value {
        json!({
            "kind": "create", "planning": "melody", "style": "rock", "lyrics": "[verse]\nla",
            "seed": 7, "cfgScale": 1.01, "steps": 32, "tier": "q4", "decoder": "legacy",
            "scoreSampling": { "maxTokens": 2048 },
            "semanticSampling": { "minTokens": 16000, "maxTokens": 16000 }
        })
    }

    /// The capture renders exactly the job the API would queue for the same song: the same
    /// validated `Yue2JobSpec`, and — through the job path's own `build_request` — the same engine
    /// request (sampling, decoder, guidance, steps, artifacts, memory) and the same admission pins.
    #[test]
    fn the_capture_request_is_the_job_paths_request() {
        let job: Yue2JobSpec = serde_json::from_value(full_job_body()).unwrap();
        contract::validate_request(&job).unwrap();
        let spec = case_spec(&full_case()).unwrap();
        assert_eq!(spec, job);

        let run_dir = Path::new("/tmp/yue2-profile/run");
        let captured = case_request(&full_case(), run_dir).unwrap();
        let queued = build_request(&job, &Inputs::default(), run_dir, CancelFlag::new());
        assert_eq!(captured.prompt, queued.prompt);
        assert_eq!(captured.seed, queued.seed);
        assert_eq!(captured.steps, queued.steps);
        assert_eq!(captured.guidance, queued.guidance);
        assert_eq!(
            format!("{:?}", captured.audio),
            format!("{:?}", queued.audio)
        );
        assert_eq!(
            format!("{:?}", captured.memory),
            format!("{:?}", queued.memory)
        );
        assert_eq!(
            format!("{:?}", admission_pins(&spec).unwrap()),
            format!("{:?}", admission_pins(&job).unwrap())
        );
        // And the request really carries the case (not two equally empty requests).
        let song = captured.audio.as_ref().unwrap().song.clone().unwrap();
        assert_eq!(song.planning, Some(SongPlanning::Melody));
        assert_eq!(song.decoder, Some(SongDecoder::Legacy));
        assert_eq!(song.score_sampling.unwrap().max_tokens, Some(2048));
        assert_eq!(captured.steps, Some(32));
        assert_eq!(captured.guidance, Some(1.01));
    }

    /// Every case the checked-in plan declares is a job the API accepts.
    #[test]
    fn every_planned_case_is_a_valid_job() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../config/yue2-memory-profile-plan.json"
        ))
        .unwrap();
        let requests = plan["requests"].as_object().unwrap();
        assert!(!requests.is_empty());
        for (name, spec) in requests {
            for tier in spec["tiers"].as_array().unwrap() {
                let backend = spec["backends"]
                    .as_array()
                    .and_then(|rows| rows.first())
                    .and_then(Value::as_str)
                    .unwrap_or("metal");
                let parsed = case(json!({
                    "id": format!("yue2:{}:{backend}:{name}", tier.as_str().unwrap()),
                    "tier": tier,
                    "decoder": spec["decoder"],
                    "arMode": spec.get("arMode"),
                    "request": spec["request"],
                }))
                .unwrap();
                case_spec(&parsed).unwrap_or_else(|why| panic!("{name}: {why}"));
            }
        }
    }

    /// A result.json shaped like a published song run's (`candle_audio_yue2::run`).
    fn run_result() -> Value {
        json!({
            "schema": "yue2-run-v1", "kind": "song", "status": "complete",
            "identity": "run-identity", "plan_identity": "plan-identity",
            "truncated": { "abc": false, "semantic": true },
            "decoder": { "release": "legacy", "repo": "m-a-p/YuE2-Vae-legacy" },
            "latent": { "sha256": "a".repeat(64), "shape": [100, 64], "dtype": "float32" },
            "timing": { "nar_seconds": 3.5, "vae_seconds": 1.25, "e2e_seconds": 20.0 },
        })
    }

    /// Each stage holds the time from its mark to the next; `done` closes the last stage.
    #[test]
    fn stage_wall_times_run_from_each_mark_to_the_next() {
        let seconds = stage_seconds(&[
            ("load", 10.0),
            ("plan", 12.5),
            ("semantic", 20.0),
            ("acoustic", 30.0),
            ("decode", 34.0),
            ("done", 35.5),
        ]);
        let got: Vec<(&str, f64)> = seconds
            .iter()
            .map(|(stage, secs)| (stage.as_str(), secs.as_f64().unwrap()))
            .collect();
        assert_eq!(
            got,
            [
                ("acoustic", 4.0),
                ("decode", 1.5),
                ("load", 2.5),
                ("plan", 7.5),
                ("semantic", 10.0)
            ]
        );
        assert!(!seconds.contains_key("done"));
    }

    /// outcome.json carries the run's truncation flags, identities and timings as the run stated
    /// them, and refuses a run record that does not state one.
    #[test]
    fn the_outcome_carries_the_runs_truncation_identities_and_timings() {
        let marks = stage_seconds(&[("load", 1.0), ("plan", 2.0), ("done", 4.0)]);
        let outcome = outcome_json(
            "case",
            42.0,
            0.2,
            &run_result(),
            OutcomeModes {
                ar_mode: contract::ArMode::Native,
                compute_policy: None,
                backend: crate::yue2_admission::Yue2Backend::Cpu,
                engine_config: None,
            },
            marks,
        )
        .unwrap();
        assert_eq!(
            outcome["truncated"],
            json!({ "abc": false, "semantic": true })
        );
        assert_eq!(outcome["runIdentity"], "run-identity");
        assert_eq!(outcome["planIdentity"], "plan-identity");
        assert_eq!(outcome["decoder"]["release"], "legacy");
        assert_eq!(outcome["latent"]["sha256"], "a".repeat(64));
        assert_eq!(outcome["engineTiming"]["vae_seconds"], 1.25);
        assert_eq!(outcome["stageSeconds"], json!({ "load": 1.0, "plan": 2.0 }));
        assert_eq!(outcome["audioSeconds"], 42.0);

        for (pointer, broken) in [
            ("/truncated", Value::Null),
            ("/truncated/semantic", json!("yes")),
            ("/latent/sha256", json!("short")),
            ("/decoder", Value::Null),
            ("/timing", Value::Null),
            ("/identity", Value::Null),
        ] {
            let mut result = run_result();
            *result.pointer_mut(pointer).unwrap() = broken;
            assert!(
                outcome_json(
                    "case",
                    42.0,
                    0.2,
                    &result,
                    OutcomeModes {
                        ar_mode: contract::ArMode::Native,
                        compute_policy: None,
                        backend: crate::yue2_admission::Yue2Backend::Cpu,
                        engine_config: None
                    },
                    Default::default(),
                )
                .is_err(),
                "a run record with a broken {pointer} must be refused, not defaulted"
            );
        }
    }

    #[test]
    fn fp8_outcome_requires_the_engines_effective_fp8_config() {
        let result = run_result();
        let capture = |config: Option<&Value>| {
            outcome_json(
                "case",
                42.0,
                0.2,
                &result,
                OutcomeModes {
                    ar_mode: contract::ArMode::ExperimentalFp8,
                    compute_policy: None,
                    backend: crate::yue2_admission::Yue2Backend::Cuda,
                    engine_config: config,
                },
                Default::default(),
            )
        };
        let fp8 = json!({ "quantization": "fp8" });
        assert_eq!(capture(Some(&fp8)).unwrap()["engineQuantization"], "fp8");
        assert!(capture(None).is_err());
        assert!(capture(Some(&json!({ "quantization": "none" }))).is_err());
    }

    #[test]
    fn explicit_precision_case_binds_job_and_effective_stage_dtypes() {
        let mut case = full_case();
        case.compute_policy = Some(contract::ComputePolicy::Bf16);
        let spec = case_spec(&case).unwrap();
        assert_eq!(spec.compute_policy, Some(contract::ComputePolicy::Bf16));
        contract::validate_new_submission(&spec).unwrap();
        let config = json!({
            "compute_policy": "bf16", "model_dtype": "bfloat16", "vae_dtype": "bfloat16",
        });
        let capture = |config: &Value| {
            outcome_json(
                &case.id,
                42.0,
                0.2,
                &run_result(),
                OutcomeModes {
                    ar_mode: contract::ArMode::Native,
                    compute_policy: case.compute_policy,
                    backend: crate::yue2_admission::Yue2Backend::Cpu,
                    engine_config: Some(config),
                },
                Default::default(),
            )
        };
        let outcome = capture(&config).unwrap();
        assert_eq!(outcome["engineComputePolicy"], "bf16");
        assert_eq!(outcome["engineModelDtype"], "bfloat16");
        assert_eq!(outcome["engineVaeDtype"], "bfloat16");
        assert!(capture(
            &json!({ "compute_policy": "bf16", "model_dtype": "bfloat16", "vae_dtype": "float32" })
        )
        .is_err());
    }

    #[test]
    fn effective_vae_math_policy_is_bound_to_the_selected_backend_and_precision() {
        use crate::yue2_admission::Yue2Backend;

        let config = |policy: &str, dtype: &str, math: Option<&str>| {
            let mut config = json!({
                "compute_policy": policy,
                "model_dtype": dtype,
                "vae_dtype": dtype,
            });
            if let Some(math) = math {
                config["vae_cuda_bf16_math_policy"] = json!(math);
            }
            config
        };
        let outcome = |backend, compute_policy, config: &Value| {
            outcome_json(
                "case",
                42.0,
                0.2,
                &run_result(),
                OutcomeModes {
                    ar_mode: contract::ArMode::Native,
                    compute_policy: Some(compute_policy),
                    backend,
                    engine_config: Some(config),
                },
                Default::default(),
            )
        };
        let selected = "fixed_order_bf16_convolution_v1";
        let cuda_bf16 = config("bf16", "bfloat16", Some(selected));
        assert_eq!(
            outcome(Yue2Backend::Cuda, contract::ComputePolicy::Bf16, &cuda_bf16).unwrap()
                ["engineVaeCudaBf16MathPolicy"],
            selected
        );
        for invalid in [
            config("bf16", "bfloat16", None),
            config("bf16", "bfloat16", Some("stale_or_unknown")),
            config(
                "bf16",
                "bfloat16",
                Some("disallow_reduced_precision_reduction_v1"),
            ),
        ] {
            assert!(outcome(Yue2Backend::Cuda, contract::ComputePolicy::Bf16, &invalid).is_err());
        }
        for backend in [Yue2Backend::Cpu, Yue2Backend::Metal] {
            assert!(outcome(backend, contract::ComputePolicy::Bf16, &cuda_bf16).is_err());
            assert!(outcome(
                backend,
                contract::ComputePolicy::Bf16,
                &config("bf16", "bfloat16", None)
            )
            .unwrap()
            .get("engineVaeCudaBf16MathPolicy")
            .is_none());
        }
        for backend in [Yue2Backend::Cuda, Yue2Backend::Cpu, Yue2Backend::Metal] {
            for policy in [contract::ComputePolicy::Fp32, contract::ComputePolicy::Auto] {
                let mut effective = if policy == contract::ComputePolicy::Fp32 {
                    config("fp32", "float32", None)
                } else {
                    let mut auto = config("auto", "bfloat16", None);
                    auto["vae_dtype"] = json!("float32");
                    auto
                };
                assert!(outcome(backend, policy, &effective)
                    .unwrap()
                    .get("engineVaeCudaBf16MathPolicy")
                    .is_none());
                effective["vae_cuda_bf16_math_policy"] = json!(selected);
                assert!(outcome(backend, policy, &effective).is_err());
            }
        }
    }

    /// The plan's case shape maps onto the job's request; unknown fields and values are refused.
    #[test]
    fn a_case_file_renders_the_job_shaped_request() {
        let parsed = case(json!({
            "id": "yue2:q4:metal:long-context",
            "tier": "q4",
            "decoder": "legacy",
            "request": {
                "planning": "off", "style": "rock", "lyrics": "[verse]\nla", "seed": 7,
                "cfgScale": 1.01,
                "semanticSampling": { "minTokens": 16000, "maxTokens": 16000 }
            }
        }))
        .expect("parses");
        let request = case_request(&parsed, Path::new("/tmp/run")).unwrap();
        let song = request.audio.as_ref().unwrap().song.clone().unwrap();
        assert_eq!(song.planning, Some(SongPlanning::Off));
        assert_eq!(song.decoder, Some(SongDecoder::Legacy));
        assert_eq!(song.semantic_sampling.unwrap().max_tokens, Some(16000));
        assert_eq!(request.guidance, Some(1.01));
        assert!(case(
            json!({ "id": "x", "tier": "q4", "decoder": "standard", "extra": 1,
            "request": { "planning": "full", "style": "", "lyrics": "", "seed": 1 } })
        )
        .is_err());
        let bad = case(json!({ "id": "x", "tier": "q4", "decoder": "loud",
            "request": { "planning": "full", "style": "", "lyrics": "", "seed": 1 } }))
        .unwrap();
        assert!(case_request(&bad, Path::new("/tmp/run")).is_err());
    }

    /// The acoustic prefill and solve are one measured stage whose estimate is the larger of the two
    /// on each pool.
    #[test]
    fn the_record_estimate_merges_the_acoustic_phases() {
        use crate::yue2_admission::{
            estimate, shape_of, Yue2ArMode, Yue2Backend, Yue2Controls, Yue2Precision,
        };
        let request = case_request(
            &case(json!({ "id": "x", "tier": "q8", "decoder": "standard",
                "request": { "planning": "full", "style": "pop", "lyrics": "[verse]\nla", "seed": 1 } }))
            .unwrap(),
            Path::new("/tmp/run"),
        )
        .unwrap();
        let load = Yue2LoadFacts {
            tier: Yue2Tier::Q8,
            precision: Yue2Precision::Default,
            ar: Yue2ArMode::Native,
            sequential_offload: false,
        };
        let shape = shape_of(&builtin_entry(), &request, load, None, Yue2ArMode::Native).unwrap();
        let est = estimate(&shape, Yue2Backend::Metal, Yue2Controls::production(), None).unwrap();
        let json = estimate_json(&est);
        let stages = json["stages"].as_object().unwrap();
        let keys: Vec<&str> = stages.keys().map(String::as_str).collect();
        assert_eq!(keys, ["acoustic", "decode", "load", "plan", "semantic"]);
        let acoustic = [Yue2Stage::AcousticPrefill, Yue2Stage::AcousticSolve]
            .map(|s| est.stage(s).unwrap().total_bytes())
            .into_iter()
            .max()
            .unwrap();
        assert_eq!(stages["acoustic"]["totalBytes"], acoustic);
        assert_eq!(json["evidence"], "estimate_pending_terminal_calibration");
    }
}
