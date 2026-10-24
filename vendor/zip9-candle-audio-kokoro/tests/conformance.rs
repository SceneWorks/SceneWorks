//! Real-weight conformance for the candle Kokoro provider (sc-12836) — the epic's DoD gate:
//! real pinned weights → registry load-by-id → generate → a real, speech-like WAV.
//!
//! Both tests are `#[ignore]`d and snapshot-gated like every other family's real-weight tests:
//! set `KOKORO_SNAPSHOT` to a `hexgrad/Kokoro-82M` snapshot dir (config.json +
//! kokoro-v1_0.pth + voices/), or leave it unset to resolve the pinned snapshot through the
//! audio lane's F-029 hub path (downloads ~330 MB into the ordinary HF cache on first run).
//!
//! ```text
//! cargo test --locked -p candle-audio-kokoro --test conformance -- --ignored --nocapture
//! ```
//!
//! `kokoro_wav_conformance` also writes the synthesized WAV next to the test output
//! (`KOKORO_WAV_OUT` overrides the path) so a human can listen to the evidence.

use std::path::PathBuf;

use candle_audio_kokoro::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use candle_audio_kokoro::{candle_audio, pipeline};

/// Resolve the snapshot from the required `KOKORO_SNAPSHOT` env (a passed-in `hexgrad/Kokoro-82M`
/// snapshot dir). Inference never self-fetches or derives a cache location (epic 13657); the
/// real-weights harness points this at a pre-materialized snapshot.
fn snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(std::env::var("KOKORO_SNAPSHOT").expect(
        "set KOKORO_SNAPSHOT to a hexgrad/Kokoro-82M snapshot dir (config.json + kokoro-v1_0.pth + voices/)",
    )))
}

/// The shared **audio-generator** conformance suite (sc-12853): audio validate-honesty (a valid
/// audio request accepted; unadvertised voice / language / sample-rate + visual conditioning rejected
/// as typed errors), a well-formed `AudioTrack` output, progress + progress contract, typed mid-run
/// and pre-generate cancellation, and seed determinism — against the real provider, resolved
/// **through the explicit registry**. This replaces the earlier ad-hoc use of the *image* `Generator`
/// suite (`gen_core_testkit::conformance` / `Profile`), whose oversize-rejection check is meaningless
/// for a `Modality::Audio` model; `audio_conformance` is purpose-built for it.
#[test]
#[ignore = "real weights: needs a hexgrad/Kokoro-82M snapshot (KOKORO_SNAPSHOT); run with --ignored"]
fn kokoro_conformance() {
    let spec = LoadSpec::new(snapshot());
    let profile = gen_core_testkit::AudioProfile {
        prompt: "The quick brown fox jumps over the lazy dog.".to_owned(),
        // Kokoro folds synthesis into a fixed 5-stage bar (pipeline::STAGES).
        steps: pipeline::STAGES,
        seed: 42,
        cancel_steps: pipeline::STAGES,
        // Empty audio sub-block: Kokoro resolves its default voice (af_heart), so the positive
        // request stays inside the advertised surface.
        audio: Default::default(),
    };
    gen_core_testkit::audio_conformance(
        || {
            candle_audio_kokoro::provider_registry()
                .unwrap()
                .load(candle_audio_kokoro::MODEL_ID, &spec)
                .expect("kokoro_82m loads through the explicit registry")
        },
        &profile,
    );
}

/// The real-WAV DoD: fixed script + seed + voice → non-empty, 24 kHz, mono, finite,
/// NON-SILENT speech-shaped audio of plausible duration, written to disk as a playable WAV.
#[test]
#[ignore = "real weights: needs a hexgrad/Kokoro-82M snapshot (KOKORO_SNAPSHOT); run with --ignored"]
fn kokoro_wav_conformance() {
    let spec = LoadSpec::new(snapshot());
    let registry = candle_audio_kokoro::provider_registry().unwrap();
    let generator = registry
        .load(candle_audio_kokoro::MODEL_ID, &spec)
        .expect("kokoro_82m loads through the explicit registry");
    assert_eq!(generator.descriptor().id, "kokoro_82m");

    let req = GenerationRequest {
        prompt: "The quick brown fox jumps over the lazy dog.".into(),
        seed: Some(42),
        audio: Some(AudioParams {
            voice: Some("af_heart".into()),
            language: Some("en".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut steps = 0u32;
    let mut decoding = 0u32;
    let out = generator
        .generate(&req, &mut |p| match p {
            Progress::Step { .. } => steps += 1,
            Progress::Decoding => decoding += 1,
            Progress::Loading(_) => {}
        })
        .expect("generate");
    assert_eq!(steps, pipeline::STAGES);
    assert_eq!(decoding, 1);

    let track = match out {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected GenerationOutput::Audio, got {other:?}"),
    };
    assert_eq!(track.sample_rate, 24_000);
    assert_eq!(track.channels, 1, "Kokoro is mono");
    assert!(!track.samples.is_empty());
    assert!(
        track.samples.iter().all(|s| s.is_finite()),
        "finite samples"
    );

    // Duration: a plausible band for the 9-word script at natural speed.
    let secs = track.samples.len() as f32 / track.sample_rate as f32;
    assert!(
        (2.0..=8.0).contains(&secs),
        "duration {secs:.2}s outside the plausible 2-8 s band"
    );

    // NON-SILENT: overall RMS over the interior (skips boundary sentinels' lead-in/out).
    let n = track.samples.len();
    let interior = &track.samples[n / 10..n - n / 10];
    let rms = (interior.iter().map(|s| s * s).sum::<f32>() / interior.len() as f32).sqrt();
    assert!(rms > 0.01, "interior RMS {rms:.5} — silence is a failure");

    // Speech-shaped energy over time: 50 ms frame RMS must VARY (voiced peaks vs pauses),
    // not sit flat like a tone or noise floor.
    let frame_len = 1200; // 50 ms
    let frame_rms: Vec<f32> = track
        .samples
        .chunks(frame_len)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let max_frame = frame_rms.iter().cloned().fold(0.0f32, f32::max);
    let mean_frame = frame_rms.iter().sum::<f32>() / frame_rms.len() as f32;
    let var_frame = frame_rms
        .iter()
        .map(|r| (r - mean_frame) * (r - mean_frame))
        .sum::<f32>()
        / frame_rms.len() as f32;
    let cv = var_frame.sqrt() / mean_frame;
    assert!(
        max_frame > 0.05,
        "peak frame RMS {max_frame:.4} too weak for speech"
    );
    assert!(
        cv > 0.25,
        "frame-RMS coefficient of variation {cv:.3} — speech has voiced peaks and pauses, \
         constant energy is not speech"
    );

    // VOICED PERIODICITY: the highest-energy 50 ms window must autocorrelate strongly at a
    // plausible pitch lag (70–400 Hz → lags 60–343 at 24 kHz). This is what an
    // amplitude-modulated rumble or shaped noise cannot fake: real voiced speech is
    // quasi-periodic at the speaker's F0 (the af_heart reference run measures r ≈ 0.86 at
    // ~316 Hz; 0.4 leaves generous headroom without admitting noise).
    let best = frame_rms
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    let start = best * frame_len;
    let seg = &track.samples[start..(start + 2 * frame_len).min(track.samples.len())];
    let mean = seg.iter().sum::<f32>() / seg.len() as f32;
    let seg: Vec<f32> = seg.iter().map(|s| s - mean).collect();
    let r0: f32 = seg.iter().map(|s| s * s).sum();
    assert!(r0 > 0.0, "voiced window has no energy");
    let mut best_r = 0.0f32;
    let mut best_lag = 0usize;
    for lag in 60..=343usize {
        let r: f32 = seg[..seg.len() - lag]
            .iter()
            .zip(&seg[lag..])
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / r0;
        if r > best_r {
            best_r = r;
            best_lag = lag;
        }
    }
    assert!(
        best_r > 0.4,
        "voiced-window autocorrelation peak {best_r:.3} (lag {best_lag} ≈ {:.0} Hz) below 0.4 — \
         no pitch periodicity means this is not voiced speech",
        24_000.0 / best_lag.max(1) as f32
    );

    // Spectral tilt: speech concentrates energy in the low band. Compare sub-4 kHz vs
    // supra-8 kHz energy via the shared radix-2 STFT (n_fft 512 @ 24 kHz → 46.9 Hz bins).
    let window = candle_audio::dsp::hann_window(512);
    let spec = candle_audio::dsp::stft(interior, 512, 256, &window).expect("stft");
    let mag = spec.magnitude();
    let (mut low, mut high) = (0.0f64, 0.0f64);
    for bin in 0..spec.n_bins {
        let hz = bin as f32 * 24_000.0 / 512.0;
        let e: f64 = mag[bin * spec.n_frames..(bin + 1) * spec.n_frames]
            .iter()
            .map(|m| (*m as f64) * (*m as f64))
            .sum();
        if hz < 4_000.0 {
            low += e;
        } else if hz >= 8_000.0 {
            high += e;
        }
    }
    assert!(
        low > 10.0 * high,
        "sub-4 kHz energy ({low:.1}) must dominate supra-8 kHz ({high:.1}) for speech"
    );

    // Write the playable evidence.
    let out_path = std::env::var("KOKORO_WAV_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("kokoro-sc12836.wav"));
    candle_audio::wav::write_wav_pcm16(&out_path, &track).expect("write WAV");
    println!(
        "kokoro_wav_conformance: wrote {} ({secs:.2}s, RMS {rms:.4}, peak frame RMS \
         {max_frame:.4}, pitch autocorr {best_r:.3} @ {:.0} Hz)",
        out_path.display(),
        24_000.0 / best_lag.max(1) as f32,
    );

    // Determinism at the WAV layer too: the same request+seed re-synthesizes byte-identically.
    let out2 = generator
        .generate(&req, &mut |_| {})
        .expect("second generate");
    let track2 = match out2 {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    };
    assert_eq!(
        track.samples, track2.samples,
        "seeded synthesis must be deterministic"
    );
}
