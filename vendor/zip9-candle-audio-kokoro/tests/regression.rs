//! Kokoro-82M regression fixture (sc-12854) — the audio validation harness
//! (`candle_audio::harness`) run against the real provider, asserted against the committed
//! metric envelope + PCM repeatability hash in
//! `tests/fixtures/kokoro_82m_regression.json`.
//!
//! `#[ignore]`d and snapshot-gated exactly like `tests/conformance.rs`: set
//! `KOKORO_SNAPSHOT` to a `hexgrad/Kokoro-82M` snapshot dir, or leave it unset to resolve
//! the pinned snapshot through the audio lane's F-029 hub path.
//!
//! ```text
//! cargo test --locked -p candle-audio-kokoro --test regression -- --ignored --nocapture
//! ```
//!
//! What the fixture pins (numbers only — generated media stays out of git):
//!
//! - **Envelope drift fails**: every harness run's duration, integrated loudness (LUFS),
//!   true peak (dBTP), clipping count, sample rate, and channel count must sit inside the
//!   committed [`MetricEnvelope`] bands.
//! - **Repeatability**: all runs in the process must produce byte-identical PCM, and on the
//!   fixture's canonical platform (os/arch recorded in the fixture — the same class as the
//!   real-weights runner) the SHA-256 must equal the committed hash exactly. On other
//!   platforms the exact-hash check is skipped (Candle CPU kernels are not bit-identical
//!   across architectures) and the observed hash is printed instead; the envelope and
//!   intra-process repeatability still gate.
//!
//! The exact-hash tuple is (backend, os, arch, **opt-level**) — sc-17004. Measured on the
//! real-weights runner: an opt-level 0 and an opt-level 3 build of this workspace's own crates
//! produce two different PCM streams, each internally deterministic (12x opt0, 4x opt3) and
//! acoustically indistinguishable (identical duration/LUFS/dBTP/clipping). The specific codegen
//! transform responsible has **not** been isolated — notably it is *not* FMA contraction, which
//! rustc leaves off (verified: `a * b + c` emits separate `fmul`/`fadd` at `-C opt-level=3`).
//! So the fixture commits **both** hashes (`pcm_sha256_opt0`, `pcm_sha256_opt3`) and the test
//! asserts the one matching this build, keeping the gate at full strength in either profile
//! instead of going quiet in one. CI runs `--release`.
//!
//! [`MetricEnvelope`]: candle_audio::harness::MetricEnvelope

use std::path::PathBuf;

use candle_audio_kokoro::candle_audio::harness;
use candle_audio_kokoro::gen_core::{
    self, AudioParams, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};

const FIXTURE: &str = include_str!("fixtures/kokoro_82m_regression.json");

/// Resolve the snapshot from the required `KOKORO_SNAPSHOT` env (a passed-in `hexgrad/Kokoro-82M`
/// snapshot dir). Inference never self-fetches or derives a cache location (epic 13657).
fn snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(std::env::var("KOKORO_SNAPSHOT").expect(
        "set KOKORO_SNAPSHOT to a hexgrad/Kokoro-82M snapshot dir (config.json + kokoro-v1_0.pth + voices/)",
    )))
}

/// Pull a required field out of the fixture JSON, with a path-labeled panic on schema drift.
fn field<'a>(v: &'a serde_json::Value, path: &[&str]) -> &'a serde_json::Value {
    let mut cur = v;
    for key in path {
        cur = cur
            .get(key)
            .unwrap_or_else(|| panic!("fixture is missing `{}`", path.join(".")));
    }
    cur
}

#[test]
#[ignore = "real weights: needs a hexgrad/Kokoro-82M snapshot (KOKORO_SNAPSHOT); run with --ignored"]
fn kokoro_regression_fixture() {
    let fx: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON parses");
    assert_eq!(
        field(&fx, &["model"]).as_str(),
        Some(candle_audio_kokoro::MODEL_ID),
        "fixture targets this provider"
    );
    let runs = field(&fx, &["runs"]).as_u64().expect("runs") as usize;
    let envelope = harness::MetricEnvelope {
        sample_rate: field(&fx, &["envelope", "sample_rate"])
            .as_u64()
            .expect("sample_rate") as u32,
        channels: field(&fx, &["envelope", "channels"])
            .as_u64()
            .expect("channels") as u16,
        min_duration_secs: field(&fx, &["envelope", "min_duration_secs"])
            .as_f64()
            .expect("min_duration_secs"),
        max_duration_secs: field(&fx, &["envelope", "max_duration_secs"])
            .as_f64()
            .expect("max_duration_secs"),
        min_integrated_lufs: field(&fx, &["envelope", "min_integrated_lufs"])
            .as_f64()
            .expect("min_integrated_lufs") as f32,
        max_integrated_lufs: field(&fx, &["envelope", "max_integrated_lufs"])
            .as_f64()
            .expect("max_integrated_lufs") as f32,
        max_true_peak_dbtp: field(&fx, &["envelope", "max_true_peak_dbtp"])
            .as_f64()
            .expect("max_true_peak_dbtp") as f32,
        max_clipped_samples: field(&fx, &["envelope", "max_clipped_samples"])
            .as_u64()
            .expect("max_clipped_samples") as usize,
    };

    // Load through the explicit registry, exactly like conformance.
    let spec = LoadSpec::new(snapshot());
    let generator = candle_audio_kokoro::provider_registry()
        .unwrap()
        .load(candle_audio_kokoro::MODEL_ID, &spec)
        .expect("kokoro_82m loads through the explicit registry");

    let req = GenerationRequest {
        prompt: field(&fx, &["script"]).as_str().expect("script").to_owned(),
        seed: Some(field(&fx, &["seed"]).as_u64().expect("seed")),
        audio: Some(AudioParams {
            voice: Some(field(&fx, &["voice"]).as_str().expect("voice").to_owned()),
            language: Some(
                field(&fx, &["language"])
                    .as_str()
                    .expect("language")
                    .to_owned(),
            ),
            ..Default::default()
        }),
        ..Default::default()
    };

    let report = harness::measure_generation(
        || match generator.generate(&req, &mut |_| {})? {
            GenerationOutput::Audio(track) => Ok(track),
            other => Err(gen_core::Error::Msg(format!(
                "expected GenerationOutput::Audio, got {other:?}"
            ))),
        },
        runs,
    )
    .expect("harness completes every run");

    // Envelope: any drift on any run fails, with every finding named.
    for (i, metrics) in report.runs.iter().enumerate() {
        let violations = envelope.violations(metrics);
        assert!(
            violations.is_empty(),
            "run {i} drifted outside the committed envelope:\n  {}",
            violations.join("\n  ")
        );
        // The deferred quality slots must stay honest until sc-12851 (CLAP) / sc-12850
        // (ASR) exist — a Some here means someone faked a score.
        assert_eq!(metrics.prompt_adherence, None, "CLAP not landed (sc-12851)");
        assert_eq!(metrics.lyric_alignment, None, "ASR not landed (sc-12850)");
    }

    // Intra-process repeatability: every seeded run must be byte-identical.
    let hash = report
        .repeatability_hash()
        .expect("seeded runs produce byte-identical PCM");

    // Exact committed hash on the fixture's canonical platform (the real-weights runner).
    let fx_os = field(&fx, &["repeatability", "os"]).as_str().expect("os");
    let fx_arch = field(&fx, &["repeatability", "arch"])
        .as_str()
        .expect("arch");
    // sc-17004: the exact-hash tuple includes OPT-LEVEL. Measured on the real-weights runner, an
    // opt-level 0 and an opt-level 3 build produce different (each internally deterministic) PCM,
    // so one committed hash cannot gate both. `[profile.dev.package."*"] opt-level = 3` covers
    // registry DEPENDENCIES only — not workspace members — so candle is identical across profiles
    // and the moving parts are this workspace's own crates (`candle-audio-kokoro` and its
    // `candle-audio` host-DSP dependency, both opt-level 0 in dev). The exact transform is NOT
    // isolated, and in particular is not FMA contraction: rustc leaves FP contraction off, and
    // `a * b + c` still emits separate `fmul`/`fadd` at `-C opt-level=3` on this target.
    //
    // We therefore commit one hash per opt-level and select for this build. `debug_assertions` is a
    // CORRELATE of opt-level, not the causal variable (`CARGO_PROFILE_DEV_OPT_LEVEL=3` moves the
    // hash while leaving debug_assertions on) — it is chosen because it needs no build script and
    // is accurate for this workspace's two stock profiles, which have no `[profile.dev]` /
    // `[profile.release]` overrides. Known false-failure modes, all of them loud rather than
    // silent: an overridden opt-level (diagnosed by name below), `opt-level` 1 or 2 (never
    // baselined — reported as drift), and `[profile.release] debug-assertions = true` (selects the
    // opt0 key on an opt3 build). Reading the real level would take a `build.rs` exporting
    // `OPT_LEVEL`; that would be this workspace's first build script, which is not worth it for a
    // single fixture.
    let (fx_key, other_key, profile) = if cfg!(debug_assertions) {
        (
            "pcm_sha256_opt0",
            "pcm_sha256_opt3",
            "unoptimized (opt-level 0)",
        )
    } else {
        (
            "pcm_sha256_opt3",
            "pcm_sha256_opt0",
            "optimized (opt-level 3)",
        )
    };
    let fx_hash = field(&fx, &["repeatability", fx_key])
        .as_str()
        .expect("pcm_sha256 for this build's opt-level");
    let fx_other = field(&fx, &["repeatability", other_key])
        .as_str()
        .expect("pcm_sha256 for the other opt-level");
    // The exact PCM hash is a **CPU** determinism gate: candle's Metal/CUDA kernels are not
    // bit-identical to CPU, so the committed hash applies only to a CPU build on the canonical
    // platform. On a GPU build (this crate compiled `--features metal` / `--features cuda`) the
    // exact hash cannot match — so we skip it and let the **metric envelope** (asserted for every
    // run above: duration/LUFS/dBTP/clipping/rate/channels) plus intra-process repeatability be the
    // Metal/CUDA drift gate (sc-13928). Frame-shape on GPU is separately gated by
    // `tests/conformance.rs` (frame-RMS CV + voiced periodicity). Without this cfg scope a
    // `--features metal` run on macos/aarch64 would spuriously fail against the CPU hash.
    let cpu_build = cfg!(not(any(feature = "metal", feature = "cuda")));
    if cpu_build && std::env::consts::OS == fx_os && std::env::consts::ARCH == fx_arch {
        // The two baselines must differ, or the diagnosis below would fire on a correct build.
        // They can legitimately converge one day (a contraction-stable rewrite), and if they do
        // this fixture wants ONE hash, not two equal ones — so fail here rather than let the
        // diagnosis make every run red.
        assert_ne!(
            fx_hash, fx_other,
            "fixture baselines {fx_key} and {other_key} are identical — if the profiles have \
             genuinely converged, collapse them to a single committed hash instead"
        );
        // Name the confusable failure rather than letting it read as plain drift: landing exactly
        // on the OTHER committed baseline is far more likely a build-configuration mismatch than a
        // coincidental regression. It is not provably so from the hash alone, so say both.
        assert_ne!(
            hash, fx_other,
            "PCM hash matches the fixture's {other_key} baseline but this build is {profile}. \
             Most likely the build's real opt-level does not match its profile default (an \
             overridden opt-level, e.g. CARGO_PROFILE_DEV_OPT_LEVEL, or `debug-assertions` set \
             against the profile default) — in which case the output is a known-good baseline and \
             nothing has drifted. If the build is stock, then the two profiles have converged, \
             which is a real change in the seeded output and wants a fixture re-baseline."
        );
        assert_eq!(
            hash, fx_hash,
            "PCM repeatability hash drifted from the committed fixture on the canonical \
             platform ({fx_os}/{fx_arch}, {profile}) — the seeded output changed. Only opt-level \
             0 and 3 are baselined; if this build uses opt-level 1 or 2, that alone explains the \
             mismatch and is not a regression."
        );
    } else {
        let backend = if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(feature = "metal") {
            "metal"
        } else {
            "cpu"
        };
        println!(
            "note: exact-hash check skipped (backend={backend}, platform {}/{} vs fixture \
             canonical cpu {fx_os}/{fx_arch}); metric envelope + intra-process repeatability gate \
             this run — observed hash {hash} for a {profile} build (the hash is opt-level \
             dependent, so quote the profile alongside it)",
            std::env::consts::OS,
            std::env::consts::ARCH,
        );
    }

    // The measured evidence, for envelope review and re-baselining.
    let first = &report.runs[0];
    println!(
        "kokoro_regression_fixture: {runs} runs | first-run latency {:.3}s, steady {:?}, \
         warmup overhead {:?} | peak RSS {} | duration {:.3}s @ {} Hz x{} | \
         {:.2} LUFS, {:.2} dBTP, {} clipped | hash {hash}",
        first.latency.as_secs_f64(),
        report.steady_latency(),
        report.warmup_overhead(),
        first
            .peak_rss_bytes
            .map_or("unavailable".to_owned(), |b| format!(
                "{:.1} MiB",
                b as f64 / (1024.0 * 1024.0)
            )),
        first.duration_secs,
        first.sample_rate,
        first.channels,
        first.integrated_lufs,
        first.true_peak_dbtp,
        first.clipped_samples,
    );
}
