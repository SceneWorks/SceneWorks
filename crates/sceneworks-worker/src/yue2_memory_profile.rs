//! One fresh-process YuE2 memory-profile capture (sc-23001) — the native half of
//! `scripts/yue2-memory-profile.mjs`, which the terminal story (sc-23002) runs ONCE over the cases in
//! `config/yue2-memory-profile-plan.json`. Nothing here runs in an ordinary test run.
//!
//! The Node controller owns the case list, repository and closure-digest identity, the memory
//! sampler (Darwin: the `memory-calibration-watchdog.py` footprint guard; CUDA: `nvidia-smi`) and
//! the record. This module owns the reviewable native work, through the SAME seams a YuE2 job uses:
//! the installed weights resolved as the job resolves them (the pinned bf16 snapshot, or the locally
//! derived q8 / q4 tier verified against its pins), the job's admission gate
//! ([`crate::yue2_admission::check`], against this host's live budget), and the registered `yue2`
//! provider loaded through [`crate::inference_runtime::load_audio`]. It writes, into the case's output
//! directory:
//!
//! * `admission.json` — the admission outcome: refused (with the gate's message), or the chosen
//!   controls, the per-stage estimate and its evidence class;
//! * `stages.jsonl` — `{"stage", "at"}` marks (wall-clock seconds, the sampler's clock) as the
//!   render moves through load → plan → semantic → acoustic → decode, driven by the admission lease
//!   from the engine's own progress, so the controller attributes each memory sample to a stage;
//! * `outcome.json` — completed / failed, with the audio's length and RMS.
//!
//! ```text
//! SCENEWORKS_ENABLE_YUE2_MEMORY_PROFILE=1 SCENEWORKS_YUE2_PROFILE_CASE=<case.json> \
//!   SCENEWORKS_YUE2_PROFILE_OUT=<dir> SCENEWORKS_DATA_DIR=<app data dir> \
//!   cargo test --release -p sceneworks-worker --lib yue2_memory_profile::capture_case -- \
//!   --ignored --exact --nocapture --test-threads 1
//! ```

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use gen_core::{
    AudioArtifacts, AudioParams, GenerationOutput, GenerationRequest, LoadSpec, Precision,
    Progress, Quant, SongDecoder, SongParams, SongPlanning, TokenSampling, WeightsSource,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::yue2_admission::{Yue2LoadFacts, Yue2Stage, Yue2Tier};
use crate::Settings;

const ENABLE_ENV: &str = "SCENEWORKS_ENABLE_YUE2_MEMORY_PROFILE";
const CASE_ENV: &str = "SCENEWORKS_YUE2_PROFILE_CASE";
const OUT_ENV: &str = "SCENEWORKS_YUE2_PROFILE_OUT";

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
    cfg_scale: Option<f32>,
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
    request: CaseRequest,
}

fn sampling(s: &CaseSampling) -> TokenSampling {
    TokenSampling {
        min_tokens: s.min_tokens,
        max_tokens: s.max_tokens,
        ..Default::default()
    }
}

/// The job-shaped request a case renders into `run_dir`.
pub(crate) fn case_request(
    case: &ProfileCase,
    run_dir: &Path,
) -> Result<GenerationRequest, String> {
    let planning = match case.request.planning.as_str() {
        "full" => SongPlanning::Full,
        "melody" => SongPlanning::Melody,
        "off" => SongPlanning::Off,
        other => return Err(format!("unknown planning {other:?}")),
    };
    let decoder = match case.decoder.as_str() {
        "standard" => SongDecoder::Standard,
        "legacy" => SongDecoder::Legacy,
        other => return Err(format!("unknown decoder {other:?}")),
    };
    Ok(GenerationRequest {
        prompt: case.request.style.clone(),
        seed: Some(case.request.seed),
        steps: case.request.steps,
        guidance: case.request.cfg_scale,
        audio: Some(AudioParams {
            lyrics: Some(case.request.lyrics.clone()),
            song: Some(SongParams {
                planning: Some(planning),
                decoder: Some(decoder),
                score_sampling: case.request.score_sampling.as_ref().map(sampling),
                semantic_sampling: case.request.semantic_sampling.as_ref().map(sampling),
                ..Default::default()
            }),
            artifacts: Some(AudioArtifacts {
                dir: run_dir.to_path_buf(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    })
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
    last: Option<&'static str>,
}

impl StageMarks {
    fn open(path: &Path) -> Self {
        Self {
            file: std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("open stage marks"),
            last: None,
        }
    }

    fn mark(&mut self, stage: &'static str) {
        if self.last == Some(stage) {
            return;
        }
        self.last = Some(stage);
        let line = json!({ "stage": stage, "at": now_secs() });
        writeln!(self.file, "{line}").expect("write stage mark");
        self.file.flush().expect("flush stage mark");
    }
}

/// The installed weights directory for `tier`, resolved exactly as the YuE2 job resolves it.
fn tier_dir(data_dir: &Path, entry: &Value, tier: Yue2Tier) -> PathBuf {
    use sceneworks_core::model_artifacts::artifact_selection::{
        derived_snapshot_state, local_derivation, local_derivation_snapshot_dir,
        DerivedSnapshotState,
    };
    let row = entry["downloads"]
        .as_array()
        .expect("downloads")
        .iter()
        .find(|row| row["variant"] == tier.key())
        .unwrap_or_else(|| panic!("the catalog declares no {} tier", tier.key()));
    match local_derivation(row) {
        None => crate::model_jobs::huggingface_pinned_snapshot_dir(
            data_dir,
            row["repo"].as_str().expect("repo"),
            row["revision"].as_str().expect("revision"),
        )
        .unwrap_or_else(|| {
            panic!(
                "the {} weights are not installed under {data_dir:?}",
                tier.key()
            )
        }),
        Some(derivation) => {
            let dir = local_derivation_snapshot_dir(data_dir, "yue2", tier.key(), &derivation)
                .expect("a safe derivation path");
            assert!(
                matches!(
                    derived_snapshot_state(&dir, &derivation),
                    DerivedSnapshotState::Verified
                ),
                "the derived {} tier at {dir:?} is not installed and verified",
                tier.key()
            );
            dir
        }
    }
}

fn component_dir(data_dir: &Path, entry: &Value, component: &str) -> Option<PathBuf> {
    let row = entry["downloads"]
        .as_array()?
        .iter()
        .find(|row| row["componentId"] == component)?;
    crate::model_jobs::huggingface_pinned_snapshot_dir(
        data_dir,
        row["repo"].as_str()?,
        row["revision"].as_str()?,
    )
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
    let tier = Yue2Tier::from_key(&case.tier).expect("a YuE2 tier");
    let settings = Settings::from_env();
    let entry = builtin_entry();
    let run_dir = out.join("run");
    let mut request = case_request(&case, &run_dir).expect("case request");

    let weights = tier_dir(&settings.data_dir, &entry, tier);
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights));
    spec.precision = Precision::default();
    spec.quantize = match tier {
        Yue2Tier::Bf16 => None,
        Yue2Tier::Q8 => Some(Quant::Q8),
        Yue2Tier::Q4 => Some(Quant::Q4),
    };
    for (component, id) in [("vae", "vae"), ("vae_legacy", "vae_legacy")] {
        if let Some(dir) = component_dir(&settings.data_dir, &entry, component) {
            spec = spec.with_component(id, WeightsSource::Dir(dir));
        }
    }

    crate::yue2_admission::probe_hardware_in_this_test();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let admitted = runtime.block_on(crate::yue2_admission::check(
        "yue2",
        &entry,
        &request,
        Yue2LoadFacts::of(tier, &spec),
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
    if let Some(memory) = admitted.memory {
        request.memory = Some(memory);
    }
    let mut lease = admitted.lease;
    let mut marks = StageMarks::open(&out.join("stages.jsonl"));
    marks.mark("load");
    let generator = crate::inference_runtime::load_audio("yue2", &spec).expect("load yue2");
    let report = {
        let mut on_progress = |progress: Progress| {
            lease.observe(&progress);
            marks.mark(stage_key(lease.stage()));
        };
        generator
            .generate_with_report(&request, &mut on_progress)
            .expect("render")
    };
    // The generator's memory goes first, then the lease that held it (the job's drop order).
    drop(generator);
    drop(lease);
    marks.mark("done");
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
    write_json(
        &out.join("outcome.json"),
        &json!({ "caseId": case.id, "outcome": "completed", "audioSeconds": seconds, "rms": rms }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(json: Value) -> Result<ProfileCase, serde_json::Error> {
        serde_json::from_value(json)
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
            sequential_offload: false,
        };
        let shape = shape_of(&builtin_entry(), &request, load, Yue2ArMode::Native).unwrap();
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
