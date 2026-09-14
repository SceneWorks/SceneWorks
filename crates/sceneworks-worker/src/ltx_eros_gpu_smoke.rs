//! SC-18902 evidence harness for the former Windows/CUDA `ltx_2_3_eros` route.
//!
//! This is deliberately an ignored, real-weight test rather than product routing. It captured the
//! rejected Candle baseline (the Eros checkpoint with no distill adapter) at the shipped defaults.
//! It does not offer a candidate: the complete cond_safe LoRA has 3,320 keys while the current
//! Candle adapter surface accepts only 768, so filtering it would produce misleading evidence.
//! The artifact showed unusable noise and informed removal of the Candle route; this harness retains
//! the exact evidence request without making the unsupported route reachable in product code.
//!
//! The workflow pins every remote artifact revision and fixes every request parameter here. Only
//! local paths and the output directory come from the environment. In particular there is no
//! prompt/seed/shape override that could change the evidence request.

use std::path::{Path, PathBuf};
use std::time::Instant;

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use serde_json::{json, Value};

use super::smoke_support::{
    image_mean, image_std, mean_abs_frame_delta, save_png, DEGENERATE_STD_FLOOR_DEFAULT,
};
use super::video_jobs::{write_wav_pcm16, AudioTrack};

const STORY_ID: &str = "SC-18902";
const MODEL_ID: &str = "ltx_2_3_eros";
const ENGINE_ID: &str = "ltx_2_3_distilled";
const BACKEND: &str = "candle-cuda";
const CAPTURE_KIND: &str = "current-candle-route-baseline";

const WIDTH: u32 = 768;
const HEIGHT: u32 = 512;
const DURATION_SECONDS: u32 = 6;
const FPS: u32 = 25;
const STEPS: u32 = 8;
const RAW_FRAMES: u32 = DURATION_SECONDS * FPS;
const RENDERED_FRAMES: u32 = 153;
const SEED: u64 = 18_902;
const PROMPT: &str = "A wide locked camera shot of a red fox walking steadily from left to right through a snowy pine clearing at sunrise. The same fox remains visible and anatomically consistent for the entire shot. Fine snow falls continuously; no cuts, no text.";

const EROS_REPOSITORY: &str = "TenStrip/LTX2.3-10Eros";
const EROS_REVISION: &str = "84a05a13610d78dbe4340d1be23fd8185e10f697";
const EROS_FILE: &str = "10Eros_v1.3_bf16.safetensors";
const GEMMA_REPOSITORY: &str = "SceneWorks/ltx-2.3-mlx";
const GEMMA_REVISION: &str = "01df27d308466533aa09d251e3aebdcc627d07eb";
const GEMMA_SUBDIR: &str = "gemma";

fn required_env_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("set ${name}"));
    let trimmed = value.trim();
    assert!(!trimmed.is_empty(), "${name} must not be empty");
    PathBuf::from(trimmed)
}

fn assert_file(path: &Path, label: &str) {
    assert!(
        path.is_file(),
        "{label} is not a file at {}",
        path.display()
    );
}

fn assert_dir(path: &Path, label: &str) {
    assert!(
        path.is_dir(),
        "{label} is not a directory at {}",
        path.display()
    );
}

fn load_spec(eros_dir: PathBuf, gemma_dir: PathBuf) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(eros_dir)).with_text_encoder(WeightsSource::Dir(gemma_dir))
}

fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        width: WIDTH,
        height: HEIGHT,
        count: 1,
        seed: Some(SEED),
        steps: Some(STEPS),
        frames: Some(RENDERED_FRAMES),
        fps: Some(FPS),
        ..Default::default()
    }
}

fn audio_stats(audio: &gen_core::AudioTrack) -> Value {
    assert!(audio.sample_rate > 0, "LTX returned zero audio sample rate");
    assert!(audio.channels > 0, "LTX returned zero audio channels");
    assert!(
        !audio.samples.is_empty(),
        "LTX returned an empty audio track"
    );
    assert!(
        audio.samples.iter().all(|sample| sample.is_finite()),
        "LTX audio contains a non-finite sample"
    );
    assert_eq!(
        audio.samples.len() % audio.channels as usize,
        0,
        "LTX audio is not whole interleaved sample frames"
    );
    let mean = audio
        .samples
        .iter()
        .map(|sample| *sample as f64)
        .sum::<f64>()
        / audio.samples.len() as f64;
    let std = (audio
        .samples
        .iter()
        .map(|sample| (*sample as f64 - mean).powi(2))
        .sum::<f64>()
        / audio.samples.len() as f64)
        .sqrt();
    let peak = audio
        .samples
        .iter()
        .fold(0.0_f32, |current, sample| current.max(sample.abs()));
    assert!(peak > 0.0, "LTX returned an all-zero audio track");
    json!({
        "present": true,
        "sampleRate": audio.sample_rate,
        "channels": audio.channels,
        "sampleCount": audio.samples.len(),
        "durationSeconds": audio.samples.len() as f64
            / (audio.sample_rate as f64 * audio.channels as f64),
        "mean": mean,
        "std": std,
        "peak": peak,
    })
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn manifest_default_request_is_fixed_and_ltx_lattice_correct() {
        assert_eq!((WIDTH, HEIGHT), (768, 512));
        assert_eq!((DURATION_SECONDS, FPS, STEPS), (6, 25, 8));
        assert_eq!(RAW_FRAMES, 150);
        assert_eq!(
            sceneworks_core::video_request::ltx_frame_count(RAW_FRAMES),
            RENDERED_FRAMES
        );
        assert_eq!(SEED, 18_902);
        let request = request();
        assert_eq!(request.prompt, PROMPT);
        assert_eq!(request.frames, Some(153));
        assert_eq!(request.fps, Some(25));
        assert_eq!(request.steps, Some(8));
        assert_eq!(request.sampler, None);
        assert_eq!(request.guidance, None);
        assert_eq!(request.negative_prompt, None);
    }

    #[test]
    fn load_spec_is_the_current_product_neutral_no_adapter_baseline() {
        let spec = load_spec(PathBuf::from("eros"), PathBuf::from("gemma"));
        let WeightsSource::Dir(weights) = &spec.weights else {
            panic!("baseline Eros weights must remain a snapshot directory")
        };
        assert_eq!(weights, &PathBuf::from("eros"));
        let Some(WeightsSource::Dir(text_encoder)) = &spec.text_encoder else {
            panic!("baseline Gemma weights must remain a snapshot directory")
        };
        assert_eq!(text_encoder, &PathBuf::from("gemma"));
        assert!(spec.adapters.is_empty());
    }
}

#[test]
#[ignore = "real-weight SC-18902 acceptance; needs exact public Eros/Gemma artifacts and a Windows CUDA GPU"]
fn ltx_eros_candle_gpu_smoke() {
    let eros_dir = required_env_path("LTX_EROS_DIR");
    let gemma_dir = required_env_path("LTX_EROS_GEMMA_DIR");
    let out_dir = required_env_path("LTX_EROS_OUT_DIR");
    let eros_file = eros_dir.join(EROS_FILE);
    assert_dir(&eros_dir, "exact Eros snapshot");
    assert_file(&eros_file, "Eros bf16 checkpoint");
    assert_dir(&gemma_dir, "public SceneWorks Gemma component");
    assert_file(&gemma_dir.join("config.json"), "Gemma config");

    std::fs::create_dir_all(out_dir.join("frames")).expect("create frame output directory");
    std::fs::create_dir_all(out_dir.join("stills")).expect("create still output directory");

    println!(
        "[sc-18902] capture={} model={} engine={} {}x{} duration={}s raw_frames={} rendered_frames={} fps={} steps={} seed={} prompt={:?}",
        CAPTURE_KIND,
        MODEL_ID,
        ENGINE_ID,
        WIDTH,
        HEIGHT,
        DURATION_SECONDS,
        RAW_FRAMES,
        RENDERED_FRAMES,
        FPS,
        STEPS,
        SEED,
        PROMPT
    );

    let load_started = Instant::now();
    let generator = crate::inference_runtime::load(ENGINE_ID, &load_spec(eros_dir, gemma_dir))
        .unwrap_or_else(|error| panic!("load Candle LTX Eros provider: {error}"));
    let load_seconds = load_started.elapsed().as_secs_f64();
    assert_eq!(generator.descriptor().id, ENGINE_ID);

    let render_started = Instant::now();
    let mut last_step = 0_u32;
    let mut saw_decode = false;
    let output = generator
        .generate(&request(), &mut |progress| match progress {
            Progress::Step { current, total } => {
                assert!(current >= last_step, "LTX progress moved backwards");
                assert!(current <= total, "LTX progress exceeded its total");
                last_step = current;
                println!("[sc-18902] denoise {current}/{total}");
            }
            Progress::Decoding => {
                saw_decode = true;
                println!("[sc-18902] decoding");
            }
            Progress::Loading(phase) => println!("[sc-18902] loading {phase:?}"),
        })
        .unwrap_or_else(|error| panic!("Candle LTX Eros render: {error}"));
    let render_seconds = render_started.elapsed().as_secs_f64();
    assert!(last_step > 0, "LTX emitted no denoise progress");
    assert!(saw_decode, "LTX emitted no decode progress");

    let GenerationOutput::Video { frames, fps, audio } = output else {
        panic!("Candle LTX Eros returned a non-video output")
    };
    assert_eq!(fps, FPS, "Candle LTX changed the requested playback fps");
    assert_eq!(
        frames.len(),
        RENDERED_FRAMES as usize,
        "Candle LTX changed the duration-derived frame count"
    );

    let mut frame_stats = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(
            (frame.width, frame.height),
            (WIDTH, HEIGHT),
            "frame {index} has the wrong dimensions"
        );
        assert_eq!(
            frame.pixels.len(),
            (WIDTH * HEIGHT * 3) as usize,
            "frame {index} has an invalid RGB buffer"
        );
        let mean = image_mean(frame);
        let std = image_std(frame);
        assert!(
            std > DEGENERATE_STD_FLOOR_DEFAULT,
            "frame {index} is degenerate (std {std:.3})"
        );
        save_png(
            frame,
            &out_dir.join("frames").join(format!("frame_{index:05}.png")),
        );
        frame_stats.push(json!({ "index": index, "mean": mean, "std": std }));
    }

    let pair_deltas: Vec<f64> = frames
        .windows(2)
        .map(|pair| mean_abs_frame_delta(&pair[0], &pair[1]))
        .collect();
    let minimum_pair_delta = pair_deltas.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        minimum_pair_delta > 0.0,
        "Candle LTX returned at least one bit-identical consecutive frame pair"
    );
    let first_last_delta = mean_abs_frame_delta(&frames[0], &frames[frames.len() - 1]);

    for (name, index) in [
        ("first", 0_usize),
        ("middle", frames.len() / 2),
        ("last", frames.len() - 1),
    ] {
        save_png(
            &frames[index],
            &out_dir.join("stills").join(format!("{name}.png")),
        );
    }

    let audio = audio.expect("Candle LTX Eros must return its synchronized audio track");
    let audio_metadata = audio_stats(&audio);
    write_wav_pcm16(
        &AudioTrack {
            samples: audio.samples,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        },
        &out_dir.join("audio.wav"),
    )
    .expect("write synchronized LTX audio");

    let adapter_reports = generator
        .adapter_apply_reports()
        .into_iter()
        .map(|report| {
            json!({
                "path": report.adapter_path,
                "applied": report.applied,
                "skipped": report.skipped,
            })
        })
        .collect::<Vec<_>>();
    assert!(
        adapter_reports.is_empty(),
        "the current-route baseline must not install or apply an adapter"
    );
    let metadata = json!({
        "schemaVersion": 1,
        "story": STORY_ID,
        "model": MODEL_ID,
        "engine": ENGINE_ID,
        "backend": BACKEND,
        "captureKind": CAPTURE_KIND,
        "request": {
            "prompt": PROMPT,
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "durationSeconds": DURATION_SECONDS,
            "fps": FPS,
            "steps": STEPS,
            "sampler": null,
            "guidance": null,
            "negativePrompt": null,
            "rawFrames": RAW_FRAMES,
            "renderedFrames": RENDERED_FRAMES,
            "renderedFramesDurationSeconds": RENDERED_FRAMES as f64 / FPS as f64,
            "productDurationSeconds": DURATION_SECONDS,
        },
        "artifacts": {
            "checkpoint": {
                "repository": EROS_REPOSITORY,
                "revision": EROS_REVISION,
                "file": EROS_FILE,
            },
            "textEncoder": {
                "repository": GEMMA_REPOSITORY,
                "revision": GEMMA_REVISION,
                "subdir": GEMMA_SUBDIR,
            },
        },
        "result": {
            "frameCount": frames.len(),
            "fps": fps,
            "loadSeconds": load_seconds,
            "renderSeconds": render_seconds,
            "minimumPairDelta": minimum_pair_delta,
            "firstLastDelta": first_last_delta,
            "pairDeltas": pair_deltas,
            "frameStats": frame_stats,
            "audio": audio_metadata,
            "adapterApplyReports": adapter_reports,
        },
    });
    std::fs::write(
        out_dir.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata).expect("serialize SC-18902 metadata"),
    )
    .expect("write SC-18902 metadata");

    println!(
        "[sc-18902] DONE capture={} load={load_seconds:.1}s render={render_seconds:.1}s min_pair_delta={minimum_pair_delta:.3} first_last_delta={first_last_delta:.3} out={}",
        CAPTURE_KIND,
        out_dir.display()
    );
}
