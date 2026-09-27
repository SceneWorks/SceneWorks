//! sc-23001: YuE2 residency model, admission choice/refusal, and residency leases.
//!
//! Every budget below is DERIVED from the estimator itself (a stage's own bytes, ± 1), never a
//! machine's number, so these assert the model's structure and the decision's logic rather than
//! any measurement. The one set of absolute figures asserted — sc-22995's recorded weight
//! residencies — is content-derived (tensor shapes × storage), identical on every machine.

use gen_core::{
    AudioParams, GenerationRequest, SavedPlan, SongCover, SongCoverMode, SongDecoder, SongParams,
    SongPlanning, TokenSampling,
};
use serde_json::json;

use super::*;

/// Serializes the tests that open leases or run [`check`]: the live-lease table is process-wide, so
/// a lease another test holds would otherwise shrink a unified budget under a parallel test. Async
/// tests hold it around their own current-thread runtime (never across an `.await`).
static LEASE_TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn lease_test_serial() -> std::sync::MutexGuard<'static, ()> {
    LEASE_TEST_SERIAL
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Run `future` on a fresh current-thread runtime (so the thread-local budget override and lease
/// bookkeeping stay on this thread).
pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(future)
}

pub(crate) fn builtin_yue2_entry() -> Value {
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

fn round_gb(bytes: u64) -> f64 {
    (bytes as f64 / 1e7).round() / 100.0
}

fn residency(tier: Yue2Tier, backend: Yue2Backend) -> Yue2WeightResidency {
    weight_residency(
        tier,
        backend,
        Yue2Precision::Default,
        Yue2ArMode::Native,
        None,
    )
    .expect("a supported combination")
}

fn load(tier: Yue2Tier) -> Yue2LoadFacts {
    Yue2LoadFacts {
        tier,
        precision: Yue2Precision::Default,
        sequential_offload: false,
    }
}

fn default_request() -> GenerationRequest {
    GenerationRequest {
        prompt: "indie pop, female vocal".into(),
        audio: Some(AudioParams {
            lyrics: Some("[verse]\nla la la\n[chorus]\noh oh oh".into()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn shape(tier: Yue2Tier, request: &GenerationRequest) -> Yue2Shape {
    shape_of(
        &builtin_yue2_entry(),
        request,
        load(tier),
        None,
        Yue2ArMode::Native,
    )
    .expect("the request prices")
}

fn with_song(song: SongParams) -> GenerationRequest {
    let mut request = default_request();
    request.audio.as_mut().unwrap().song = Some(song);
    request
}

fn metal(capacity_bytes: u64) -> Yue2Budget {
    Yue2Budget::Unified {
        backend: Yue2Backend::Metal,
        capacity_bytes,
        resident_bytes: 0,
        reclaimable_bytes: 0,
    }
}

fn cuda(free_bytes: u64, total_bytes: u64) -> Yue2Budget {
    Yue2Budget::Dedicated {
        free_bytes,
        total_bytes,
        reclaimable_bytes: 0,
        host_available_bytes: None,
        gpu_id: "0".into(),
        compute_cap: Some(12.0),
    }
}

fn priced(shape: &Yue2Shape, backend: Yue2Backend, controls: Yue2Controls) -> Yue2Estimate {
    estimate(shape, backend, controls, Some(12.0)).expect("prices")
}

fn admitted(decision: Yue2Admission) -> Yue2Estimate {
    match decision {
        Yue2Admission::Admit(estimate) => estimate,
        other => panic!("expected Admit, got {other:?}"),
    }
}

fn refused(decision: Yue2Admission) -> String {
    match decision {
        Yue2Admission::Refuse(WorkerError::InvalidPayload(message)) => message,
        other => panic!("expected an InvalidPayload refusal, got {other:?}"),
    }
}

// ---- Weights --------------------------------------------------------------------------------------

/// The shape-derived residency reproduces sc-22995's recorded figures (`precision.rs`' table) on
/// every backend and tier. Mutation: Q8_0's block bytes 34 → 32, or Q4_K's 144 → 128, or dropping
/// the CUDA row padding — each moves at least one row off its recorded value.
#[test]
fn weights_reproduce_the_recorded_sc_22995_residencies() {
    use Yue2Backend::*;
    use Yue2Tier::*;
    assert_eq!(round_gb(residency(Q8, Cpu).device_bytes), 5.12);
    assert_eq!(round_gb(residency(Q4, Cpu).device_bytes), 3.52);
    assert_eq!(
        round_gb(residency(Bf16, Cpu).device_bytes),
        14.52,
        "CPU holds BF16 as F32"
    );
    assert_eq!(round_gb(residency(Bf16, Cuda).device_bytes), 7.26);
    assert_eq!(round_gb(residency(Q8, Cuda).device_bytes), 4.26);
    assert_eq!(round_gb(residency(Q4, Cuda).device_bytes), 2.66);
    let fp8 = weight_residency(
        Bf16,
        Cuda,
        Yue2Precision::Default,
        Yue2ArMode::Fp8,
        Some(8.9),
    )
    .expect("sm_89 bf16 supports FP8");
    assert_eq!(round_gb(fp8.device_bytes), 5.85);
    assert_eq!(
        round_gb(fp8.host_bytes),
        2.82,
        "the retained BF16 AR originals"
    );
    // The FP8 layout is the AR stages'; the acoustic stage restores the native one.
    assert_eq!(
        fp8.restored_device_bytes,
        residency(Bf16, Cuda).device_bytes
    );
    // Precision Fp32 on an accelerator is the F32 residency.
    let f32 = weight_residency(Bf16, Cuda, Yue2Precision::Fp32, Yue2ArMode::Native, None).unwrap();
    assert_eq!(round_gb(f32.device_bytes), 14.52);
}

/// Each tier's stored payload is the catalog's own derived weights file, less only the safetensors
/// header (q8 / q4: `localDerivation.weightsBytes`; bf16: the released file inside the row's
/// estimate). Mutation: store the NAR heads BF16 at q4 (or count a norm twice) — the payload drifts
/// by megabytes and leaves the header window.
#[test]
fn stored_tier_bytes_are_the_catalogs_weights_files() {
    let entry = builtin_yue2_entry();
    let rows = entry["downloads"].as_array().unwrap();
    for tier in [Yue2Tier::Q8, Yue2Tier::Q4] {
        let row = rows
            .iter()
            .find(|row| row["variant"] == tier.key())
            .expect("the tier row");
        let file = row["localDerivation"]["weightsBytes"].as_u64().unwrap();
        let stored = residency(tier, Yue2Backend::Metal).stored_bytes;
        assert!(
            stored < file && file - stored < 1 << 20,
            "{tier:?}: stored {stored} vs file {file}"
        );
    }
    let bf16 = rows.iter().find(|row| row["variant"] == "bf16").unwrap();
    let estimated = bf16["estimatedSizeBytes"].as_u64().unwrap();
    let stored = residency(Yue2Tier::Bf16, Yue2Backend::Metal).stored_bytes;
    assert!(stored < estimated && estimated - stored < 8 << 20);
}

/// AR offload moves exactly the AR-only weights: the embedding, `lm_head` and the AR path — never
/// the NAR twins, heads or latent table the solve needs. Mutation: classify `lm_head` as not AR-only.
#[test]
fn ar_offload_moves_only_the_ar_path() {
    for tier in Yue2Tier::ALL {
        let w = residency(tier, Yue2Backend::Cuda);
        let embedding_and_head = VOCAB * HIDDEN * 2; // embed_tokens alone, BF16 at every tier
        assert!(w.ar_only_bytes > embedding_and_head, "{tier:?}");
        // Both MoT paths are the same size, so the AR-only share is strictly under the total and
        // over the NAR-side remainder less the latent table.
        assert!(w.ar_only_bytes < w.restored_device_bytes, "{tier:?}");
        assert!(
            w.restored_device_bytes - w.ar_only_bytes > CONTEXT * HIDDEN * 2,
            "{tier:?}"
        );
    }
}

/// FP8 is refused everywhere but CUDA sm_89+ over the bf16 tier with BF16 compute — never folded
/// onto another precision, and an unread compute capability is a refusal, not a pass. Mutation:
/// drop the backend check (Metal FP8 then prices).
#[test]
fn fp8_is_refused_outside_cuda_sm89_bf16() {
    let fp8 = |tier, backend, precision, cap| {
        weight_residency(tier, backend, precision, Yue2ArMode::Fp8, cap)
    };
    use Yue2Backend::*;
    use Yue2Tier::*;
    // Refused for the backend itself, not merely for an unread compute capability.
    for backend in [Metal, Cpu] {
        for cap in [None, Some(12.0)] {
            let error = fp8(Bf16, backend, Yue2Precision::Default, cap).unwrap_err();
            assert!(error.contains("runs only on CUDA"), "{backend:?}: {error}");
        }
    }

    assert!(fp8(Q8, Cuda, Yue2Precision::Default, Some(9.0)).is_err());
    assert!(fp8(Bf16, Cuda, Yue2Precision::Fp32, Some(9.0)).is_err());
    assert!(fp8(Bf16, Cuda, Yue2Precision::Default, Some(8.6)).is_err());
    assert!(fp8(Bf16, Cuda, Yue2Precision::Default, None).is_err());
    assert!(fp8(Bf16, Cuda, Yue2Precision::Default, Some(8.9)).is_ok());
    // Through the decision: a Metal FP8 render is refused before any pricing loop.
    let fp8_shape = Yue2Shape {
        ar: Yue2ArMode::Fp8,
        ..shape(Bf16, &default_request())
    };
    let message = refused(decide("yue2", &fp8_shape, Some(&metal(u64::MAX)), 0));
    assert!(message.contains("runs only on CUDA"), "{message}");
}

/// An unknown tier key never prices as zero: it is `None`, which the callers refuse.
#[test]
fn an_unknown_tier_key_is_not_a_tier() {
    for key in ["fp8", "nvfp4", "BF16", "", "q2"] {
        assert_eq!(Yue2Tier::from_key(key), None, "{key:?}");
    }
    for tier in Yue2Tier::ALL {
        assert_eq!(Yue2Tier::from_key(tier.key()), Some(tier));
    }
}

// ---- Stages ---------------------------------------------------------------------------------------

/// One MoT serves every stage, so every generation stage holds the resident weights, and the floor
/// is the LARGEST stage, never the sum. Mutation: drop the weights term from the decode stage, or
/// make `unified_floor` sum the stages.
#[test]
fn every_stage_holds_the_mot_and_the_floor_is_the_max_not_the_sum() {
    for tier in Yue2Tier::ALL {
        for backend in [Yue2Backend::Cpu, Yue2Backend::Cuda, Yue2Backend::Metal] {
            let est = priced(
                &shape(tier, &default_request()),
                backend,
                Yue2Controls::production(),
            );
            let stages: Vec<_> = est.stages.iter().map(|s| s.stage).collect();
            assert_eq!(
                stages,
                [
                    Yue2Stage::Load,
                    Yue2Stage::Plan,
                    Yue2Stage::Semantic,
                    Yue2Stage::AcousticPrefill,
                    Yue2Stage::AcousticSolve,
                    Yue2Stage::Decode,
                ],
                "{tier:?} {backend:?}"
            );
            for stage in &est.stages {
                let weights = stage
                    .terms
                    .iter()
                    .find(|t| t.what == "resident weights")
                    .unwrap_or_else(|| panic!("{:?} holds the MoT", stage.stage));
                let expected = match stage.stage {
                    Yue2Stage::Plan | Yue2Stage::Semantic => est.weights.device_bytes,
                    _ => est.weights.restored_device_bytes,
                };
                assert_eq!(weights.device_bytes, expected, "{:?}", stage.stage);
            }
            let (_, floor) = est.unified_floor();
            let max = est.stages.iter().map(|s| s.total_bytes()).max().unwrap();
            let sum: u64 = est.stages.iter().map(|s| s.total_bytes()).sum();
            assert_eq!(floor, max);
            assert!(floor < sum);
        }
    }
}

/// Denser tiers need more at every stage; F32 CPU weights are twice the accelerator's dense part.
#[test]
fn tiers_order_the_floor_densest_first() {
    for backend in [Yue2Backend::Cpu, Yue2Backend::Cuda, Yue2Backend::Metal] {
        let floors: Vec<u64> = Yue2Tier::ALL
            .into_iter()
            .map(|tier| {
                priced(
                    &shape(tier, &default_request()),
                    backend,
                    Yue2Controls::production(),
                )
                .unified_floor()
                .1
            })
            .collect();
        assert!(
            floors[0] > floors[1] && floors[1] > floors[2],
            "{backend:?}: {floors:?}"
        );
    }
}

/// The FP8 AR mode holds its BF16 originals in host memory through the AR stages only: the
/// acoustic stage restores them (host 0, native device layout). Mutation: keep the host term on the
/// acoustic stages.
#[test]
fn fp8_originals_are_held_through_the_ar_stages_only() {
    let fp8_shape = Yue2Shape {
        ar: Yue2ArMode::Fp8,
        ..shape(Yue2Tier::Bf16, &default_request())
    };
    let est = priced(&fp8_shape, Yue2Backend::Cuda, Yue2Controls::production());
    for stage in [Yue2Stage::Plan, Yue2Stage::Semantic] {
        assert_eq!(
            est.stage(stage).unwrap().host_bytes(),
            est.weights.host_bytes
        );
    }
    for stage in [Yue2Stage::AcousticSolve, Yue2Stage::Decode] {
        assert_eq!(est.stage(stage).unwrap().host_bytes(), 0, "{stage:?}");
    }
    let native = priced(
        &shape(Yue2Tier::Bf16, &default_request()),
        Yue2Backend::Cuda,
        Yue2Controls::production(),
    );
    assert!(
        est.stage(Yue2Stage::Semantic).unwrap().device_bytes()
            < native.stage(Yue2Stage::Semantic).unwrap().device_bytes()
    );
}

fn kv_term(est: &Yue2Estimate, stage: Yue2Stage) -> u64 {
    est.stage(stage)
        .unwrap()
        .terms
        .iter()
        .filter(|t| t.what.contains("KV cache"))
        .map(|t| t.device_bytes)
        .sum()
}

/// Guidance runs two semantic KV caches (the negative branch's own), none on the ABC stage; a cache
/// never outgrows the released context. Mutation: price CFG as one cache, or drop the context cap.
#[test]
fn cfg_doubles_the_semantic_cache_and_caches_cap_at_the_context() {
    let song = |guidance: Option<f32>, semantic: Option<u32>| {
        let mut request = with_song(SongParams {
            semantic_sampling: semantic.map(|max_tokens| TokenSampling {
                max_tokens: Some(max_tokens),
                ..Default::default()
            }),
            ..Default::default()
        });
        request.guidance = guidance;
        shape(Yue2Tier::Q8, &request)
    };
    let off = priced(
        &song(Some(1.0), None),
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    let on = priced(
        &song(Some(1.5), None),
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    assert_eq!(
        kv_term(&on, Yue2Stage::Semantic),
        2 * kv_term(&off, Yue2Stage::Semantic)
    );
    assert_eq!(
        kv_term(&on, Yue2Stage::Plan),
        kv_term(&off, Yue2Stage::Plan)
    );
    let huge = priced(
        &song(Some(1.0), Some(1_000_000)),
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    assert_eq!(
        kv_term(&huge, Yue2Stage::Semantic),
        kv_cache_bytes(CONTEXT, 2)
    );
    // The chunk cache holds at most the context too.
    assert!(kv_term(&huge, Yue2Stage::AcousticSolve) <= kv_cache_bytes(CONTEXT, 2));
}

/// Longer songs cost more (until the context caps them); each control grows only its own stage.
#[test]
fn stages_are_monotone_in_the_request_and_in_their_own_control() {
    let semantic = |tokens: u32| {
        let request = with_song(SongParams {
            semantic_sampling: Some(TokenSampling {
                max_tokens: Some(tokens),
                ..Default::default()
            }),
            ..Default::default()
        });
        priced(
            &shape(Yue2Tier::Q4, &request),
            Yue2Backend::Metal,
            Yue2Controls::production(),
        )
    };
    let totals = |est: &Yue2Estimate, stage| est.stage(stage).unwrap().total_bytes();
    let mut last = None;
    for tokens in [100, 1000, 4000, 9000] {
        let est = semantic(tokens);
        let now = (
            totals(&est, Yue2Stage::Semantic),
            totals(&est, Yue2Stage::AcousticSolve),
            totals(&est, Yue2Stage::Decode),
        );
        if let Some(before) = last {
            let (a, b, c): (u64, u64, u64) = before;
            assert!(
                now.0 > a && now.1 > b && now.2 > c,
                "{tokens}: {now:?} vs {before:?}"
            );
        }
        last = Some(now);
    }
    let base = shape(Yue2Tier::Q4, &default_request());
    let production = Yue2Controls::production();
    let small_tile = Yue2Controls {
        decode_core_frames: 32,
        ..production
    };
    let small_chunk = Yue2Controls {
        attention_elements: HEADS * CONTEXT,
        ..production
    };
    let p = priced(&base, Yue2Backend::Metal, production);
    let t = priced(&base, Yue2Backend::Metal, small_tile);
    let c = priced(&base, Yue2Backend::Metal, small_chunk);
    assert!(totals(&t, Yue2Stage::Decode) < totals(&p, Yue2Stage::Decode));
    assert_eq!(
        totals(&t, Yue2Stage::AcousticSolve),
        totals(&p, Yue2Stage::AcousticSolve)
    );
    assert!(totals(&c, Yue2Stage::AcousticSolve) < totals(&p, Yue2Stage::AcousticSolve));
    assert_eq!(totals(&c, Yue2Stage::Decode), totals(&p, Yue2Stage::Decode));
    assert_eq!(
        totals(&c, Yue2Stage::AcousticPrefill),
        totals(&p, Yue2Stage::AcousticPrefill)
    );
}

/// Offload moves the AR-only weights to the host pool for the solve on CUDA, and moves nothing on a
/// unified backend (same pool). Mutation: apply the offload on Metal.
#[test]
fn offload_moves_ar_weights_host_ward_on_cuda_only() {
    let base = shape(Yue2Tier::Bf16, &default_request());
    let offload = Yue2Controls {
        offload_ar: true,
        ..Yue2Controls::production()
    };
    let resident = priced(&base, Yue2Backend::Cuda, Yue2Controls::production());
    let moved = priced(&base, Yue2Backend::Cuda, offload);
    let solve = |e: &Yue2Estimate| e.stage(Yue2Stage::AcousticSolve).unwrap().clone();
    assert_eq!(
        solve(&resident).device_bytes() - solve(&moved).device_bytes(),
        resident.weights.ar_only_bytes
    );
    assert_eq!(solve(&moved).host_bytes(), resident.weights.ar_only_bytes);
    // The prefill needs the AR path, so it is unchanged.
    assert_eq!(
        resident.stage(Yue2Stage::AcousticPrefill),
        moved.stage(Yue2Stage::AcousticPrefill)
    );
    let unified = priced(&base, Yue2Backend::Metal, offload);
    assert_eq!(
        unified.stages,
        priced(&base, Yue2Backend::Metal, Yue2Controls::production()).stages
    );
}

/// Plan-only renders only plan; a cached decode loads and decodes only; a recording's transcription
/// is its own stage — the transcriber unloads before the MoT loads, so its residency is never added
/// to the render's. Mutation: add the MoT weights to the transcription stage.
#[test]
fn partial_work_prices_only_the_stages_it_runs() {
    let stages = |request: &GenerationRequest| -> Vec<Yue2Stage> {
        priced(
            &shape(Yue2Tier::Q4, request),
            Yue2Backend::Metal,
            Yue2Controls::production(),
        )
        .stages
        .iter()
        .map(|s| s.stage)
        .collect()
    };
    assert_eq!(
        stages(&with_song(SongParams {
            plan_only: true,
            ..Default::default()
        })),
        [Yue2Stage::Load, Yue2Stage::Plan]
    );
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("latent.npy"), vec![0u8; 128 + 250 * 64 * 4]).unwrap();
    let cached = with_song(SongParams {
        cached_latents: Some(dir.path().to_path_buf()),
        ..Default::default()
    });
    assert_eq!(stages(&cached), [Yue2Stage::Load, Yue2Stage::Decode]);
    assert_eq!(
        shape(Yue2Tier::Q4, &cached).work,
        Yue2Work::DecodeCached { frames: 250 }
    );
    // A song shorter than one tile decodes in one tile of its own length
    // (`DecodeOptions::estimated_peak_bytes`). Mutation: price the full tile regardless.
    let cached_estimate = priced(
        &shape(Yue2Tier::Q4, &cached),
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    let tile = cached_estimate
        .stage(Yue2Stage::Decode)
        .unwrap()
        .terms
        .iter()
        .find(|t| t.what == "VAE decode tile activations")
        .unwrap()
        .device_bytes;
    assert_eq!(tile, 250 * DECODE_TILE_BYTES_PER_FRAME);

    let entry = builtin_yue2_entry();
    let transcription = transcription_shape(&entry, Yue2Tier::Q4, 180).unwrap();
    let est = priced(
        &transcription,
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    let stage = est.stage(Yue2Stage::Transcription).unwrap();
    assert!(stage.terms.iter().all(|t| t.what != "resident weights"));
    assert_eq!(
        stage.terms[0].device_bytes,
        transcriber_bytes(&entry).unwrap(),
        "SheetSage2 + MERT-v2 weights from the catalog"
    );
    let (_, floor) = est.unified_floor();
    let sum: u64 = est.stages.iter().map(|s| s.total_bytes()).sum();
    assert!(floor < sum);
    // A longer recording costs more, up to the fixed 300 s window.
    let longer = priced(
        &transcription_shape(&entry, Yue2Tier::Q4, 300).unwrap(),
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    let capped = priced(
        &transcription_shape(&entry, Yue2Tier::Q4, 900).unwrap(),
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    assert!(
        longer
            .stage(Yue2Stage::Transcription)
            .unwrap()
            .total_bytes()
            > stage.total_bytes()
    );
    assert_eq!(
        longer.stage(Yue2Stage::Transcription),
        capped.stage(Yue2Stage::Transcription)
    );
}

// ---- The request → shape --------------------------------------------------------------------------

/// The shape is read the way the engine reads the request: defaults, planning modes, supplied
/// scores, restored plans, overrides, and the `off` mode's default guidance.
#[test]
fn the_shape_follows_the_engines_request_mapping() {
    let default = shape(Yue2Tier::Q8, &default_request());
    assert_eq!(
        default.work,
        Yue2Work::Generate {
            planning: Yue2Planning::Sample {
                max_tokens: ABC_MAX_TOKENS_DEFAULT
            },
            semantic_max_tokens: SEMANTIC_MAX_TOKENS_DEFAULT,
            cfg: false,
        },
        "full planning defaults to guidance 1 (no CFG)"
    );
    let off = shape(
        Yue2Tier::Q8,
        &with_song(SongParams {
            planning: Some(SongPlanning::Off),
            ..Default::default()
        }),
    );
    assert_eq!(
        off.work,
        Yue2Work::Generate {
            planning: Yue2Planning::Off,
            semantic_max_tokens: SEMANTIC_MAX_TOKENS_DEFAULT,
            cfg: true,
        },
        "off defaults to guidance 1.01"
    );
    let mut off_no_cfg = with_song(SongParams {
        planning: Some(SongPlanning::Off),
        ..Default::default()
    });
    off_no_cfg.guidance = Some(1.0);
    assert!(matches!(
        shape(Yue2Tier::Q8, &off_no_cfg).work,
        Yue2Work::Generate { cfg: false, .. }
    ));
    let supplied = shape(
        Yue2Tier::Q8,
        &with_song(SongParams {
            score: Some("X:1\nK:C\nCDEF|".into()),
            score_sampling: Some(TokenSampling {
                max_tokens: Some(77),
                ..Default::default()
            }),
            semantic_sampling: Some(TokenSampling {
                max_tokens: Some(1234),
                ..Default::default()
            }),
            ..Default::default()
        }),
    );
    assert_eq!(
        supplied.work,
        Yue2Work::Generate {
            planning: Yue2Planning::Supplied { abc_tokens: 13 },
            semantic_max_tokens: 1234,
            cfg: false,
        }
    );
    let cover = shape(
        Yue2Tier::Q8,
        &with_song(SongParams {
            cover: Some(SongCover {
                mode: SongCoverMode::Melody,
                score: "X:1\nK:C\nCD|".into(),
                keep: None,
                translated_from: Some("source lyrics".into()),
            }),
            ..Default::default()
        }),
    );
    assert!(matches!(
        cover.work,
        Yue2Work::Generate {
            planning: Yue2Planning::Supplied { abc_tokens: 11 },
            ..
        }
    ));
    assert_eq!(
        cover.text_tokens,
        default.text_tokens + "source lyrics".len() as u64
    );
    let sampled = shape(
        Yue2Tier::Q8,
        &with_song(SongParams {
            score_sampling: Some(TokenSampling {
                max_tokens: Some(900),
                ..Default::default()
            }),
            ..Default::default()
        }),
    );
    assert!(matches!(
        sampled.work,
        Yue2Work::Generate {
            planning: Yue2Planning::Sample { max_tokens: 900 },
            ..
        }
    ));
    let plan_dir = tempfile::tempdir().unwrap();
    std::fs::write(plan_dir.path().join("abc_tokens.npy"), vec![0u8; 4 * 500]).unwrap();
    let restored = shape(
        Yue2Tier::Q8,
        &with_song(SongParams {
            plan: Some(SavedPlan {
                dir: plan_dir.path().to_path_buf(),
                identity: None,
            }),
            ..Default::default()
        }),
    );
    assert!(matches!(
        restored.work,
        Yue2Work::Generate {
            planning: Yue2Planning::Supplied { abc_tokens: 500 },
            ..
        }
    ));
    let legacy = shape(
        Yue2Tier::Q8,
        &with_song(SongParams {
            decoder: Some(SongDecoder::Legacy),
            ..Default::default()
        }),
    );
    let entry = builtin_yue2_entry();
    assert_eq!(
        legacy.decoder_bytes,
        decoder_bytes(&entry, SongDecoder::Legacy).unwrap()
    );
    assert_ne!(legacy.decoder_bytes, default.decoder_bytes);
}

/// What cannot be priced fails closed: a restored plan whose tokens are unreadable, a catalog entry
/// without the selected decoder. Mutation: fall back to zero ABC tokens / zero decoder bytes.
#[test]
fn an_unpriceable_request_is_an_error_not_a_zero() {
    let missing_plan = with_song(SongParams {
        plan: Some(SavedPlan {
            dir: std::path::PathBuf::from("/definitely/not/a/plan"),
            identity: None,
        }),
        ..Default::default()
    });
    let entry = builtin_yue2_entry();
    let price = |entry: &Value, request: &GenerationRequest| {
        shape_of(entry, request, load(Yue2Tier::Q4), None, Yue2ArMode::Native)
    };
    assert!(price(&entry, &missing_plan)
        .unwrap_err()
        .contains("saved plan"));
    let mut no_legacy = entry.clone();
    no_legacy["downloads"]
        .as_array_mut()
        .unwrap()
        .retain(|row| row["componentId"] != "vae_legacy");
    let legacy = with_song(SongParams {
        decoder: Some(SongDecoder::Legacy),
        ..Default::default()
    });
    assert!(price(&no_legacy, &legacy)
        .unwrap_err()
        .contains("vae_legacy"));
    assert!(price(&no_legacy, &default_request()).is_ok());
}

// ---- The decision ---------------------------------------------------------------------------------

/// A render that fits at the production controls is admitted with them, exactly at its floor.
/// Mutation: compare with `>` instead of `>=` in `fit` (the exact-floor budget then refuses).
#[test]
fn a_fitting_render_keeps_the_production_controls() {
    let s = shape(Yue2Tier::Q8, &default_request());
    let (_, floor) = priced(&s, Yue2Backend::Metal, Yue2Controls::production()).unified_floor();
    let est = admitted(decide("yue2", &s, Some(&metal(floor)), 0));
    assert_eq!(est.controls, Yue2Controls::production());
    // No reading at all: admitted under the production controls (never blocked without evidence).
    assert_eq!(
        admitted(decide("yue2", &s, None, 0)).controls,
        Yue2Controls::production()
    );
}

/// A decode-bound render gets the LARGEST tile core that fits — not the smallest, not the default.
/// Mutation: iterate the cores upward (the chosen core then drops to 1).
#[test]
fn a_decode_bound_render_gets_the_largest_tile_that_fits() {
    let s = shape(Yue2Tier::Q4, &default_request());
    let target = Yue2Controls {
        decode_core_frames: 100,
        ..Yue2Controls::production()
    };
    let at_target = priced(&s, Yue2Backend::Metal, target);
    let capacity = at_target.stage(Yue2Stage::Decode).unwrap().total_bytes();
    for stage in &at_target.stages {
        if stage.stage != Yue2Stage::Decode {
            assert!(
                stage.total_bytes() <= capacity,
                "precondition: {stage:?} fits"
            );
        }
    }
    let est = admitted(decide("yue2", &s, Some(&metal(capacity)), 0));
    assert_eq!(est.controls, target);
    assert!(est.unified_floor().1 <= capacity);
}

/// An acoustic-bound render gets the largest NAR score chunk that fits (and then the largest tile).
/// Mutation: skip the attention candidates (the render is then refused).
#[test]
fn an_acoustic_bound_render_gets_the_largest_score_chunk_that_fits() {
    let s = shape(Yue2Tier::Q4, &default_request());
    let rows16 = Yue2Controls {
        attention_elements: HEADS * 16 * CONTEXT,
        ..Yue2Controls::production()
    };
    let at = priced(&s, Yue2Backend::Metal, rows16);
    let capacity = at.stage(Yue2Stage::AcousticSolve).unwrap().total_bytes();
    for stage in [
        Yue2Stage::Load,
        Yue2Stage::Plan,
        Yue2Stage::Semantic,
        Yue2Stage::AcousticPrefill,
    ] {
        assert!(
            at.stage(stage).unwrap().total_bytes() <= capacity,
            "precondition: {stage:?}"
        );
    }
    let est = admitted(decide("yue2", &s, Some(&metal(capacity)), 0));
    assert_eq!(est.controls.attention_elements, HEADS * 16 * CONTEXT);
    assert!(
        !est.controls.offload_ar,
        "offload is never chosen on a unified pool"
    );
    assert!(est.controls.decode_core_frames < Yue2Controls::production().decode_core_frames);
    assert!(est.unified_floor().1 <= capacity);
}

/// On CUDA, a solve that stays short at every score chunk is rescued by AR offload (the AR-only
/// weights leave the card for the solve) — the same capacity on a unified pool is refused, because
/// offload moves nothing there. Mutation: never try the offload candidates.
#[test]
fn cuda_offloads_the_ar_path_when_the_solve_needs_it() {
    let s = shape(Yue2Tier::Q4, &default_request());
    let production = priced(&s, Yue2Backend::Cuda, Yue2Controls::production());
    let smallest_resident = priced(
        &s,
        Yue2Backend::Cuda,
        Yue2Controls {
            attention_elements: HEADS * CONTEXT,
            decode_core_frames: 1,
            ..Yue2Controls::production()
        },
    );
    // The card holds every stage but the resident solve at its smallest score chunk: the prefill,
    // the load and the smallest decode tile all fit, the solve does not.
    let prefill = production
        .stage(Yue2Stage::AcousticPrefill)
        .unwrap()
        .device_bytes();
    let decode = smallest_resident
        .stage(Yue2Stage::Decode)
        .unwrap()
        .device_bytes();
    let load = production.stage(Yue2Stage::Load).unwrap().device_bytes();
    let card = prefill.max(decode).max(load);
    let solve = smallest_resident
        .stage(Yue2Stage::AcousticSolve)
        .unwrap()
        .device_bytes();
    assert!(
        card < solve,
        "precondition: the resident solve binds above every other stage"
    );
    let free = card + dedicated_reserve_bytes();
    let est = admitted(decide("yue2", &s, Some(&cuda(free, free)), 0));
    assert!(est.controls.offload_ar, "{:?}", est.controls);
    assert!(est.device_peak().1 + dedicated_reserve_bytes() <= free);
    assert_eq!(
        est.stage(Yue2Stage::AcousticSolve).unwrap().host_bytes(),
        est.weights.ar_only_bytes
    );

    let unified = production
        .stage(Yue2Stage::AcousticPrefill)
        .unwrap()
        .total_bytes();
    let load = production.stage(Yue2Stage::Load).unwrap().total_bytes();
    assert!(
        load <= unified,
        "precondition: the load fits the unified pool"
    );
    let message = refused(decide("yue2", &s, Some(&metal(unified)), 0));
    assert!(
        message.contains("acoustic flow-matching solve"),
        "{message}"
    );
}

/// Over budget: refused BEFORE anything loads, naming the binding stage, the shortfall and a lighter
/// tier that fits, and saying the Metal figure is an estimate pending the terminal calibration.
/// Mutation: drop the lighter-tier alternative or the evidence sentence.
#[test]
fn an_over_budget_render_is_refused_with_stage_shortfall_and_alternatives() {
    let bf16 = shape(Yue2Tier::Bf16, &default_request());
    let q4 = shape(Yue2Tier::Q4, &default_request());
    let smallest = Yue2Controls {
        attention_elements: ATTENTION_ELEMENTS_MIN,
        decode_core_frames: 1,
        ..Yue2Controls::production()
    };
    // One byte below bf16's smallest-controls floor: bf16 cannot fit at any control, q4 can.
    let (binding, bf16_min) = priced(&bf16, Yue2Backend::Metal, smallest).unified_floor();
    let message = refused(decide("yue2", &bf16, Some(&metal(bf16_min - 1)), 0));
    for needle in [
        "yue2:",
        "bf16",
        "short",
        "q4",
        "estimate pending",
        "sc-23002",
        binding.label(),
    ] {
        assert!(message.contains(needle), "missing {needle:?}: {message}");
    }
    // Nothing fits a tiny pool: the refusal says so rather than inventing an alternative.
    let tiny = refused(decide("yue2", &q4, Some(&metal(1 << 30)), 0));
    assert!(
        tiny.contains("No smaller tier or shorter request fits"),
        "{tiny}"
    );
}

/// When a shorter song would fit, the refusal gives the semantic budget that does.
#[test]
fn a_refusal_names_the_longest_song_that_fits() {
    let q4 = shape(Yue2Tier::Q4, &default_request());
    let short = Yue2Shape {
        work: Yue2Work::Generate {
            planning: Yue2Planning::Sample {
                max_tokens: ABC_MAX_TOKENS_DEFAULT,
            },
            semantic_max_tokens: 1000,
            cfg: false,
        },
        ..q4
    };
    // The minimal controls of the 1000-token render bound its floor; that pool refuses the 9000.
    let smallest = Yue2Controls {
        attention_elements: HEADS * CONTEXT,
        decode_core_frames: 1,
        ..Yue2Controls::production()
    };
    let (_, capacity) = priced(&short, Yue2Backend::Metal, smallest).unified_floor();
    let message = refused(decide("yue2", &q4, Some(&metal(capacity)), 0));
    assert!(message.contains("semantic budget of up to"), "{message}");
    let tokens: u64 = message
        .split("semantic budget of up to ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.parse().ok())
        .expect("a token count");
    assert!((1000..9000).contains(&tokens), "{tokens}");
}

/// CUDA: a render that fits only once the cached generator's pool is reclaimed is admitted after an
/// evict; a card too small says so; a big-enough card with foreign residency says who holds it.
#[test]
fn cuda_budgets_distinguish_evict_small_card_and_foreign_residency() {
    let s = shape(Yue2Tier::Q8, &default_request());
    let (_, peak) = priced(&s, Yue2Backend::Cuda, Yue2Controls::production()).device_peak();
    let need = peak + dedicated_reserve_bytes();
    let evictable = Yue2Budget::Dedicated {
        free_bytes: need / 2,
        total_bytes: need * 2,
        reclaimable_bytes: need,
        host_available_bytes: None,
        gpu_id: "0".into(),
        compute_cap: Some(8.6),
    };
    assert!(matches!(
        decide("yue2", &s, Some(&evictable), 0),
        Yue2Admission::AdmitAfterEvict(_)
    ));
    let small = refused(decide("yue2", &s, Some(&cuda(1 << 30, 1 << 30)), 0));
    assert!(small.contains("has only ~"), "{small}");
    let held = refused(decide("yue2", &s, Some(&cuda(1 << 30, need * 2)), 0));
    assert!(
        held.contains("another process or model is holding VRAM"),
        "{held}"
    );
    assert!(held.contains("CUDA weights are measured"), "{held}");
}

/// The host pool is checked where it is read (the mapped weights file while loading, FP8's retained
/// originals through the AR stages). Mutation: skip the host check when a reading exists.
#[test]
fn cuda_checks_the_host_pool_where_it_is_read() {
    let fp8 = Yue2Shape {
        ar: Yue2ArMode::Fp8,
        ..shape(Yue2Tier::Bf16, &default_request())
    };
    let est = priced(&fp8, Yue2Backend::Cuda, Yue2Controls::production());
    let (_, host_peak) = est.host_peak();
    let budget = |host| Yue2Budget::Dedicated {
        free_bytes: u64::MAX / 4,
        total_bytes: u64::MAX / 4,
        reclaimable_bytes: 0,
        host_available_bytes: Some(host),
        gpu_id: "0".into(),
        compute_cap: Some(8.9),
    };
    assert!(matches!(
        decide("yue2", &fp8, Some(&budget(host_peak)), 0),
        Yue2Admission::Admit(_)
    ));
    let message = refused(decide("yue2", &fp8, Some(&budget(host_peak - 1)), 0));
    assert!(message.contains("host RAM"), "{message}");
}

/// Residency other live YuE2 renders still hold is charged against a unified pool.
#[test]
fn other_live_residency_shrinks_a_unified_pool() {
    let s = shape(Yue2Tier::Q8, &default_request());
    let (_, floor) = priced(&s, Yue2Backend::Metal, Yue2Controls::production()).unified_floor();
    assert!(matches!(
        decide("yue2", &s, Some(&metal(floor)), 0),
        Yue2Admission::Admit(_)
    ));
    let smallest = Yue2Controls {
        attention_elements: HEADS * CONTEXT,
        decode_core_frames: 1,
        ..Yue2Controls::production()
    };
    let (_, min_floor) = priced(&s, Yue2Backend::Metal, smallest).unified_floor();
    let message = refused(decide(
        "yue2",
        &s,
        Some(&metal(floor)),
        floor - min_floor + 1,
    ));
    assert!(
        message.contains("still held by another YuE2 render"),
        "{message}"
    );
}

// ---- Leases ---------------------------------------------------------------------------------------

/// A lease publishes the load stage, follows the engine's progress stage by stage (each AR stage's
/// Step count restarts from 1; `Decoding` marks the decode) and releases everything on drop.
/// Mutation: never advance on a restarted Step (the lease then holds the plan stage forever).
#[test]
fn a_lease_follows_the_stages_and_releases_on_drop() {
    let _serial = lease_test_serial();
    let est = priced(
        &shape(Yue2Tier::Q8, &default_request()),
        Yue2Backend::Metal,
        Yue2Controls::production(),
    );
    let held = |stage| {
        let s = est.stage(stage).unwrap();
        (s.device_bytes(), s.host_bytes())
    };
    let work = shape(Yue2Tier::Q8, &default_request()).work;
    let mut lease = Yue2Lease::open(est.clone(), work);
    let id = lease.id();
    assert!(lease_is_live(id));
    assert_eq!(lease.stage(), Yue2Stage::Load);
    assert_eq!(lease.held_bytes(), held(Yue2Stage::Load));
    let step = |current, total| Progress::Step { current, total };
    for (event, stage) in [
        (step(1, 4096), Yue2Stage::Plan),
        (step(2, 4096), Yue2Stage::Plan),
        (step(1, 9000), Yue2Stage::Semantic),
        (step(3, 9000), Yue2Stage::Semantic),
        (step(1, 64), Yue2Stage::AcousticSolve),
        (step(64, 64), Yue2Stage::AcousticSolve),
        (Progress::Decoding, Yue2Stage::Decode),
    ] {
        lease.observe(&event);
        assert_eq!(lease.stage(), stage, "{event:?}");
    }
    assert_eq!(lease.held_bytes(), held(Yue2Stage::Decode));
    // Earlier stages' caches were released on the way: the decode holds no KV cache.
    assert_eq!(kv_term(lease.estimate(), Yue2Stage::Decode), 0);
    drop(lease);
    assert!(!lease_is_live(id), "a dropped lease holds nothing");
}

/// The acoustic stage holds the larger of its prefill and solve phases; with AR offload on CUDA the
/// AR-only weights are held in the host pool while it runs.
#[test]
fn an_offloaded_lease_holds_the_ar_weights_host_side_during_the_acoustic_stage() {
    let _serial = lease_test_serial();
    let est = priced(
        &shape(Yue2Tier::Bf16, &default_request()),
        Yue2Backend::Cuda,
        Yue2Controls {
            offload_ar: true,
            ..Yue2Controls::production()
        },
    );
    let ar_only = est.weights.ar_only_bytes;
    let work = shape(Yue2Tier::Bf16, &default_request()).work;
    let mut lease = Yue2Lease::open(est, work);
    lease.observe(&Progress::Step {
        current: 1,
        total: ABC_MAX_TOKENS_DEFAULT as u32,
    });
    lease.observe(&Progress::Step {
        current: 1,
        total: SEMANTIC_MAX_TOKENS_DEFAULT as u32,
    });
    lease.observe(&Progress::Step {
        current: 1,
        total: 32,
    });
    assert_eq!(lease.stage(), Yue2Stage::AcousticSolve);
    assert_eq!(lease.held_bytes().1, ar_only);
    lease.observe(&Progress::Decoding);
    assert_eq!(lease.held_bytes().1, 0, "restored before the decode");
}

/// A cached decode's lease goes straight from the load to the decode.
#[test]
fn a_decode_only_lease_has_only_the_decode_stage() {
    let _serial = lease_test_serial();
    let s = Yue2Shape {
        work: Yue2Work::DecodeCached { frames: 500 },
        ..shape(Yue2Tier::Q4, &default_request())
    };
    let mut lease = Yue2Lease::open(
        priced(&s, Yue2Backend::Metal, Yue2Controls::production()),
        s.work,
    );
    lease.observe(&Progress::Decoding);
    assert_eq!(lease.stage(), Yue2Stage::Decode);
}

/// A resumed run reuses its finished stages without reporting them, so a restarted count is matched
/// to its stage by its total: a run resuming at the semantic stage is held at the semantic stage's
/// residency, not the plan's. Mutation: advance strictly by order (the lease then reads "plan").
#[test]
fn a_resumed_run_is_placed_by_its_step_totals() {
    let _serial = lease_test_serial();
    let s = shape(Yue2Tier::Q8, &default_request());
    let mut lease = Yue2Lease::open(
        priced(&s, Yue2Backend::Metal, Yue2Controls::production()),
        s.work,
    );
    lease.observe(&Progress::Step {
        current: 1,
        total: SEMANTIC_MAX_TOKENS_DEFAULT as u32,
    });
    assert_eq!(lease.stage(), Yue2Stage::Semantic);
    let mut resumed_at_acoustic = Yue2Lease::open(
        priced(&s, Yue2Backend::Metal, Yue2Controls::production()),
        s.work,
    );
    resumed_at_acoustic.observe(&Progress::Step {
        current: 1,
        total: 32,
    });
    assert_eq!(resumed_at_acoustic.stage(), Yue2Stage::AcousticSolve);
}

// ---- The hook ---------------------------------------------------------------------------------------

/// `check` fails closed on an unpriceable request BEFORE probing the hardware, and a live lease on a
/// unified pool is charged against the next admission until it drops.
#[test]
fn check_prices_before_probing_and_charges_live_leases() {
    let _serial = lease_test_serial();
    block_on(check_prices_before_probing_and_charges_live_leases_body());
}

async fn check_prices_before_probing_and_charges_live_leases_body() {
    let mut entry = builtin_yue2_entry();
    entry["downloads"]
        .as_array_mut()
        .unwrap()
        .retain(|row| row["componentId"] != "vae");
    let probes = budget_probes();
    let error = check(
        "yue2",
        &entry,
        &default_request(),
        load(Yue2Tier::Q8),
        None,
        "0",
    )
    .await
    .expect_err("no decoder, no price");
    assert!(error.to_string().contains("cannot price"), "{error}");
    assert_eq!(budget_probes(), probes, "refused before the budget probe");

    let entry = builtin_yue2_entry();
    let s = shape(Yue2Tier::Q8, &default_request());
    let smallest = Yue2Controls {
        attention_elements: ATTENTION_ELEMENTS_MIN,
        decode_core_frames: 1,
        ..Yue2Controls::production()
    };
    // Exactly the smallest-controls floor: one render fits, two cannot.
    let (_, floor) = priced(&s, Yue2Backend::Metal, smallest).unified_floor();
    let _budget = override_budget(Some(metal(floor)));
    let first = check(
        "yue2",
        &entry,
        &default_request(),
        load(Yue2Tier::Q8),
        None,
        "0",
    )
    .await
    .expect("fits alone");
    assert!(first.memory.tile_vae_decode && first.memory.chunk_attention);
    // The first render's lease still holds its load stage: the second no longer fits.
    let second = check(
        "yue2",
        &entry,
        &default_request(),
        load(Yue2Tier::Q8),
        None,
        "0",
    )
    .await;
    assert!(
        second
            .as_ref()
            .is_err_and(|e| e.to_string().contains("still held")),
        "{second:?}"
    );
    drop(first);
    check(
        "yue2",
        &entry,
        &default_request(),
        load(Yue2Tier::Q8),
        None,
        "0",
    )
    .await
    .expect("fits again once the first lease is released");
}

/// The memory block sets every field YuE2 reads, and only those.
#[test]
fn the_memory_block_carries_exactly_the_chosen_controls() {
    let controls = Yue2Controls {
        offload_ar: true,
        attention_elements: HEADS * 8 * CONTEXT,
        decode_core_frames: 57,
    };
    let memory = controls.generation_memory();
    assert_eq!(
        memory,
        GenerationMemory {
            stage_residency: true,
            chunk_attention: true,
            attention_chunk_size: Some((HEADS * 8 * CONTEXT) as u32),
            tile_vae_decode: true,
            decode_tile_edge: Some(57),
            ..GenerationMemory::default()
        }
    );
    // The production core is `DecodeOptions::production()`'s 224 frames (an 8 GiB tile budget).
    assert_eq!(Yue2Controls::production().decode_core_frames, 224);
    assert_eq!(
        attention_candidates().last(),
        Some(ATTENTION_ELEMENTS_MIN),
        "the smallest chunk is one query row at the full context"
    );
}

/// `/proc/meminfo`'s `MemAvailable` in bytes.
#[test]
fn meminfo_available_is_parsed_in_bytes() {
    let body =
        "MemTotal:       65536000 kB\nMemFree:         1000 kB\nMemAvailable:   32768000 kB\n";
    assert_eq!(parse_meminfo_available(body), Some(32_768_000 * 1024));
    assert_eq!(parse_meminfo_available("MemTotal: 1 kB\n"), None);
}

/// A CPU transcription is charged to the host pool until its lease drops. Mutation evidence:
/// omitting the live-lease subtraction admits the second identical closure on the same budget;
/// pricing just the weights admits the one-byte-short budget below.
#[tokio::test]
async fn transcription_admission_charges_host_and_releases_its_lease() {
    let entry = builtin_yue2_entry();
    let bytes = transcriber_bytes(&entry).unwrap();
    let needed: u64 = transcription_terms(300, bytes, Yue2Backend::Cpu)
        .iter()
        .map(|term| term.device_bytes + term.host_bytes)
        .sum();
    let budget = |capacity_bytes| Yue2Budget::Unified {
        backend: Yue2Backend::Metal,
        capacity_bytes,
        resident_bytes: 0,
        reclaimable_bytes: 0,
    };
    let too_small = override_budget(Some(budget(needed - 1)));
    assert!(check_transcription(&entry, 300, "0").await.is_err());
    drop(too_small);
    let exact = override_budget(Some(budget(needed)));
    let lease = check_transcription(&entry, 300, "0").await.unwrap();
    assert_eq!(live_residency_bytes(), (0, needed));
    assert!(check_transcription(&entry, 300, "0").await.is_err());
    drop(lease);
    assert_eq!(live_residency_bytes(), (0, 0));
    assert!(check_transcription(&entry, 300, "0").await.is_ok());
    drop(exact);
}

/// The catalog's advisory `candle.minMemoryGbByTier` floors are THIS estimator's derivation: the
/// smallest machine admission can admit the default song on — the default request at the smallest
/// controls admission can choose (1-frame decode core, one-row score chunks, AR offload on CUDA),
/// the larger of a Metal working set and a CUDA card (device + allocator reserve), in whole GiB,
/// rounded up. The scalar is the densest tier's. Mutation: change the estimator (or the manifest)
/// without the other.
#[test]
fn the_catalog_floors_are_the_estimators_smallest_admissible_default_render() {
    let entry = builtin_yue2_entry();
    let candle = &entry["candle"];
    assert_eq!(
        candle["measured"],
        json!(false),
        "no YuE2 memory is measured yet"
    );
    let smallest = |offload_ar| Yue2Controls {
        offload_ar,
        attention_elements: ATTENTION_ELEMENTS_MIN,
        decode_core_frames: 1,
    };
    let mut derived = BTreeMap::new();
    for tier in Yue2Tier::ALL {
        let s = shape(tier, &default_request());
        let (_, metal) = priced(&s, Yue2Backend::Metal, smallest(false)).unified_floor();
        let (_, device) = priced(&s, Yue2Backend::Cuda, smallest(true)).device_peak();
        let bytes = metal.max(device + dedicated_reserve_bytes());
        derived.insert(tier.key(), bytes.div_ceil(1 << 30));
    }
    for tier in Yue2Tier::ALL {
        assert_eq!(
            candle["minMemoryGbByTier"][tier.key()].as_u64(),
            Some(derived[tier.key()]),
            "{tier:?}: derived {derived:?}"
        );
    }
    assert_eq!(candle["minMemoryGb"].as_u64(), Some(derived["bf16"]));
}

// ---- Requested controls are knobs -----------------------------------------------------------------

fn with_memory(memory: GenerationMemory) -> GenerationRequest {
    GenerationRequest {
        memory: Some(memory),
        ..default_request()
    }
}

/// A request that carries its own memory block is priced exactly as sent — a large tile stays large
/// — and refused when it does not fit, with the controls that would. Mutation: let `choose` search
/// the cores even when the tile is pinned (the render is then admitted at a smaller tile).
#[test]
fn requested_memory_controls_are_honoured_not_overridden() {
    let big_tile = GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(600),
        ..GenerationMemory::default()
    };
    let pinned = shape(Yue2Tier::Q4, &with_memory(big_tile));
    assert_eq!(
        pinned.pins,
        Yue2Pins {
            offload_ar: Some(false),
            attention: Some(AttentionPin::Historical {
                chunk_attention: false
            }),
            decode: Some(DecodePin::Core(600)),
        }
    );
    let est = admitted(decide("yue2", &pinned, Some(&metal(u64::MAX / 4)), 0));
    assert_eq!(est.controls.decode_core_frames, 600);
    // Enough for the production tile, not for the requested one.
    let free = shape(Yue2Tier::Q4, &default_request());
    let (_, production_floor) =
        priced(&free, Yue2Backend::Metal, Yue2Controls::production()).unified_floor();
    let message = refused(decide("yue2", &pinned, Some(&metal(production_floor)), 0));
    assert!(
        message.contains("requested memory controls do not fit, but these do"),
        "{message}"
    );
    assert!(message.contains("decode_tile_edge 224"), "{message}");
    // `tile_vae_decode: false` is the production tiling, `chunk_attention: false` the 256-row tile.
    let defaults = shape(Yue2Tier::Q4, &with_memory(GenerationMemory::default()));
    assert_eq!(
        defaults.pins.decode_core_frames(),
        Some(default_decode_core())
    );
    assert_eq!(
        defaults.pins.attention_elements(),
        Some(ATTENTION_ELEMENTS_DEFAULT)
    );
}

/// A `Sequential` load pins AR offload on; values the engine refuses are refused here.
#[test]
fn the_load_and_invalid_controls_pin_or_refuse() {
    let sequential = Yue2LoadFacts {
        sequential_offload: true,
        ..load(Yue2Tier::Bf16)
    };
    let s = shape_of(
        &builtin_yue2_entry(),
        &default_request(),
        sequential,
        None,
        Yue2ArMode::Native,
    )
    .unwrap();
    assert_eq!(s.pins.offload_ar, Some(true));
    let est = admitted(decide(
        "yue2",
        &s,
        Some(&cuda(u64::MAX / 4, u64::MAX / 4)),
        0,
    ));
    assert!(est.controls.offload_ar);
    let price = |memory| {
        shape_of(
            &builtin_yue2_entry(),
            &with_memory(memory),
            load(Yue2Tier::Q4),
            None,
            Yue2ArMode::Native,
        )
    };
    let too_small = price(GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some((ATTENTION_ELEMENTS_MIN - 1) as u32),
        ..GenerationMemory::default()
    });
    assert!(too_small.unwrap_err().contains("cannot hold one query row"));
    let bad_tile = price(GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(4096),
        ..GenerationMemory::default()
    });
    assert!(bad_tile.unwrap_err().contains("decode tile core"));
    let f32 = shape_of(
        &builtin_yue2_entry(),
        &default_request(),
        Yue2LoadFacts {
            precision: Yue2Precision::Fp32,
            ..load(Yue2Tier::Bf16)
        },
        None,
        Yue2ArMode::Native,
    )
    .unwrap();
    let dense = priced(&f32, Yue2Backend::Metal, Yue2Controls::production());
    assert_eq!(
        round_gb(dense.weights.device_bytes),
        14.52,
        "an F32 load holds F32 weights"
    );
}

/// A request with a complete memory block keeps it: `check` sends it back exactly as given.
#[test]
fn check_leaves_a_requested_memory_block_untouched() {
    let _serial = lease_test_serial();
    block_on(async {
        let _budget = override_budget(Some(metal(u64::MAX / 4)));
        let block = GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(64),
            ..GenerationMemory::default()
        };
        let request = with_memory(block);
        let admitted = check(
            "yue2",
            &builtin_yue2_entry(),
            &request,
            load(Yue2Tier::Q4),
            None,
            "0",
        )
        .await
        .expect("fits");
        assert_eq!(admitted.memory, block);
        assert_eq!(admitted.lease.estimate().controls.decode_core_frames, 64);
    });
}

// ---- Hand-computed formula checks (sc-23001 review) ---------------------------------------------------
//
// Every expected value below is a literal worked out by hand from the pinned YuE2-3B tensor table
// (`manifests/yue2_3b.json` @ inference 11319984: 28 layers; `k_proj` [1024, 2048] with 128-wide
// heads ⇒ 8 KV heads; `q_proj` [2048, 2048] ⇒ 16 query heads) and the engine constants, never by
// the functions under test.

fn bare_shape(work: Yue2Work) -> Yue2Shape {
    Yue2Shape {
        text_tokens: 0,
        work,
        ..shape(Yue2Tier::Bf16, &default_request())
    }
}

fn term_bytes(est: &Yue2Estimate, stage: Yue2Stage, what: &str) -> u64 {
    est.stage(stage)
        .unwrap_or_else(|| panic!("{stage:?} priced"))
        .terms
        .iter()
        .find(|t| t.what == what)
        .unwrap_or_else(|| panic!("{stage:?} has a {what:?} term"))
        .device_bytes
}

/// KV cache: 28 layers × 2 (K, V) × 8 KV heads × 128 × positions × element bytes. At 1 000 positions
/// that is 114 688 000 bytes in BF16 and 229 376 000 in F32. Mutation: drop the K/V ×2 (or the layer
/// count) in `kv_cache_bytes`.
#[test]
fn kv_cache_matches_the_hand_computed_bytes() {
    let s = bare_shape(Yue2Work::PlanOnly {
        planning: Yue2Planning::Sample { max_tokens: 1000 },
    });
    let bf16 = priced(&s, Yue2Backend::Metal, Yue2Controls::production());
    assert_eq!(term_bytes(&bf16, Yue2Stage::Plan, "KV cache"), 114_688_000);
    let f32 = priced(&s, Yue2Backend::Cpu, Yue2Controls::production());
    assert_eq!(term_bytes(&f32, Yue2Stage::Plan, "KV cache"), 229_376_000);
}

/// NAR score tiles: 3 live tiles × 16 heads × rows × keys × element bytes, or 3 × the chunk budget
/// when that is smaller. A 100-frame song (no plan, no text: prefix 3) is one chunk of 102 NAR rows
/// over 3 + 100 + 1 + 102 = 206 keys ⇒ 3 × 16 × 102 × 206 × 2 = 2 017 152 bytes. At the one-row
/// budget (393 216 elements) over a 1 000-frame song ⇒ 3 × 393 216 × 2 = 2 359 296 bytes. Mutation:
/// drop `SCORE_TILES_LIVE` in `nar_score_bytes`.
#[test]
fn nar_score_tiles_match_the_hand_computed_bytes() {
    let song = |frames| {
        bare_shape(Yue2Work::Generate {
            planning: Yue2Planning::Off,
            semantic_max_tokens: frames,
            cfg: false,
        })
    };
    let whole = priced(&song(100), Yue2Backend::Metal, Yue2Controls::production());
    assert_eq!(
        term_bytes(
            &whole,
            Yue2Stage::AcousticSolve,
            "NAR attention score tiles"
        ),
        2_017_152
    );
    let one_row = Yue2Controls {
        attention_elements: 393_216,
        ..Yue2Controls::production()
    };
    let bounded = priced(&song(1000), Yue2Backend::Metal, one_row);
    assert_eq!(
        term_bytes(
            &bounded,
            Yue2Stage::AcousticSolve,
            "NAR attention score tiles"
        ),
        2_359_296
    );
    assert_eq!(ATTENTION_ELEMENTS_MIN, 393_216, "16 heads × 24 576 keys");
}

/// VAE decode tile: 28 MiB (29 360 128 bytes) per latent frame of `core + 2 × 16` halo frames, plus a
/// 1 GiB reserve that already covers the FP32 decoder (530 MB file < 1 073 741 824). Core 100 over a
/// 1 000-frame song ⇒ 132 × 29 360 128 = 3 875 536 896 bytes. Mutation: drop the halo, or charge the
/// decoder file on top of the reserve.
#[test]
fn the_decode_tile_matches_the_hand_computed_bytes() {
    let s = bare_shape(Yue2Work::Generate {
        planning: Yue2Planning::Off,
        semantic_max_tokens: 1000,
        cfg: false,
    });
    let controls = Yue2Controls {
        decode_core_frames: 100,
        ..Yue2Controls::production()
    };
    let est = priced(&s, Yue2Backend::Metal, controls);
    assert_eq!(
        term_bytes(&est, Yue2Stage::Decode, "VAE decode tile activations"),
        3_875_536_896
    );
    assert_eq!(
        term_bytes(&est, Yue2Stage::Decode, "FP32 VAE decoder + decode reserve"),
        1_073_741_824
    );
    assert_eq!(est.stage(Yue2Stage::Decode).unwrap().terms.len(), 4);
}

/// The acoustic chunk is sized with the SHORTEST prefix (a shorter real plan makes a longer chunk):
/// a 16 000-frame song gets chunks of (24 576 − 0 − 3) / 2 = 12 286 frames, 12 288 NAR rows, whose
/// untiled activations are 3 × 12 288 × 6 144 × 2 + 4 × 12 288 × 2 048 × 2 = 654 311 424 bytes —
/// even under a sampled 4 096-token plan, whose upper-bound prefix would have shrunk the chunk to
/// 10 087 frames. Mutation: size the chunk with the upper-bound prefix.
#[test]
fn the_acoustic_chunk_is_sized_for_the_shortest_prefix() {
    let s = Yue2Shape {
        text_tokens: 300,
        ..bare_shape(Yue2Work::Generate {
            planning: Yue2Planning::Sample { max_tokens: 4096 },
            semantic_max_tokens: 16_000,
            cfg: true,
        })
    };
    let est = priced(&s, Yue2Backend::Metal, Yue2Controls::production());
    assert_eq!(
        term_bytes(&est, Yue2Stage::AcousticSolve, "NAR activations"),
        654_311_424
    );
    // The cache still takes the upper-bound prefix, capped at the context.
    assert_eq!(
        term_bytes(&est, Yue2Stage::AcousticSolve, "chunk KV cache"),
        kv_cache_bytes(CONTEXT, 2)
    );
}

/// Mapped weights pages are clean file-backed pages, not Metal buffers: on Metal (and CPU) the load
/// holds only the resident weights; on CUDA the mapped file is charged to host RAM. Mutation: charge
/// `stored_bytes` on every backend.
#[test]
fn the_mapped_weights_file_costs_host_ram_on_cuda_only() {
    let s = shape(Yue2Tier::Bf16, &default_request());
    let metal = priced(&s, Yue2Backend::Metal, Yue2Controls::production());
    let load = metal.stage(Yue2Stage::Load).unwrap();
    assert_eq!(load.total_bytes(), 7_261_368_448, "bf16 weights alone");
    let cpu = priced(&s, Yue2Backend::Cpu, Yue2Controls::production());
    assert_eq!(cpu.stage(Yue2Stage::Load).unwrap().host_bytes(), 0);
    let cuda = priced(&s, Yue2Backend::Cuda, Yue2Controls::production());
    let load = cuda.stage(Yue2Stage::Load).unwrap();
    assert_eq!(load.device_bytes(), 7_261_368_448);
    assert_eq!(load.host_bytes(), cuda.weights.stored_bytes);
}

// ---- Field-by-field pins (sc-23001 review) -----------------------------------------------------------

/// A job that sets only `stageResidency` pins that one control: on a decode-bound budget the gate
/// still shrinks the decode tile, and the block it sends carries the job's residency, explicit
/// chunk attention and the chosen tile. Mutation: pin every control when any is set.
#[test]
fn an_unset_control_stays_free_when_another_is_set() {
    let pins = Yue2Pins::from_controls(Some(true), None, None, None, None, None).unwrap();
    assert_eq!(
        pins,
        Yue2Pins {
            offload_ar: Some(true),
            attention: None,
            decode: None,
        }
    );
    let s = Yue2Shape {
        pins,
        ..shape(Yue2Tier::Q4, &default_request())
    };
    let production = priced(&s, Yue2Backend::Metal, Yue2Controls::production());
    let (binding, floor) = production.unified_floor();
    assert_eq!(binding, Yue2Stage::Decode, "precondition: decode-bound");
    let est = admitted(decide("yue2", &s, Some(&metal(floor - 1)), 0));
    assert!(est.controls.decode_core_frames < Yue2Controls::production().decode_core_frames);
    let block = pins.memory_block(&est.controls);
    assert!(block.stage_residency, "the job's own control, as sent");
    assert!(block.tile_vae_decode && block.chunk_attention);
    assert_eq!(
        block.decode_tile_edge,
        Some(est.controls.decode_core_frames as u32)
    );
    assert_eq!(
        block.attention_chunk_size,
        Some(est.controls.attention_elements as u32)
    );
    // A pinned historical tile is echoed back exactly (no invented size).
    let historical = Yue2Pins::from_controls(None, None, Some(false), None, None, None).unwrap();
    let block = historical.memory_block(&Yue2Controls::production());
    assert!(!block.chunk_attention && block.attention_chunk_size.is_none());
    assert!(Yue2Pins::from_controls(None, None, None, Some(1 << 20), None, None).is_err());
    assert!(Yue2Pins::from_controls(None, None, None, None, None, Some(64)).is_err());
    // An explicit Resident offload policy pins offload off; a job's stageResidency wins over it.
    assert_eq!(
        Yue2Pins::from_controls(None, Some(false), None, None, None, None)
            .unwrap()
            .offload_ar,
        Some(false)
    );
    assert_eq!(
        Yue2Pins::from_controls(Some(true), Some(false), None, None, None, None)
            .unwrap()
            .offload_ar,
        Some(true)
    );
}

// ---- The resident MLX generator (sc-23001 review) -----------------------------------------------------

fn metal_with_resident(
    capacity_bytes: u64,
    resident_bytes: u64,
    reclaimable_bytes: u64,
) -> Yue2Budget {
    Yue2Budget::Unified {
        backend: Yue2Backend::Metal,
        capacity_bytes,
        resident_bytes,
        reclaimable_bytes,
    }
}

/// The MLX generator the cache keeps warm shares the Metal working set. A render that fits only
/// without it is admitted after an evict; one it still cannot free room for is refused naming what
/// holds the memory. Mutation: ignore `resident_bytes` in `fit` (then it is admitted outright), or
/// never return `FitsAfterEvict` for the unified pool.
#[test]
fn a_resident_mlx_generator_is_evicted_or_the_render_refused() {
    let s = shape(Yue2Tier::Q4, &default_request());
    let (_, floor) = priced(&s, Yue2Backend::Metal, Yue2Controls::production()).unified_floor();
    let resident = 6 << 30;
    // The working set holds the render only once the cached generator's 6 GiB are freed.
    let evictable = metal_with_resident(floor + resident - 1, resident, resident);
    match decide("yue2", &s, Some(&evictable), 0) {
        Yue2Admission::AdmitAfterEvict(est) => {
            assert_eq!(est.controls, Yue2Controls::production())
        }
        other => panic!("expected AdmitAfterEvict, got {other:?}"),
    }
    // The same memory held by something eviction does not free: nothing to reclaim.
    let smallest = Yue2Controls {
        attention_elements: ATTENTION_ELEMENTS_MIN,
        decode_core_frames: 1,
        ..Yue2Controls::production()
    };
    let (_, min_floor) = priced(&s, Yue2Backend::Metal, smallest).unified_floor();
    let pinned = metal_with_resident(min_floor + resident - 1, resident, 0);
    let message = refused(decide("yue2", &s, Some(&pinned), 0));
    assert!(
        message.contains("held by other models in this process"),
        "{message}"
    );
    // With the generator's bytes absent the same capacity admits outright.
    assert!(matches!(
        decide("yue2", &s, Some(&metal_with_resident(floor, 0, 0)), 0),
        Yue2Admission::Admit(_)
    ));
}

/// The Metal budget charges everything MLX holds and credits only what evicting the cached
/// generator and clearing the MLX cache frees. Mutation: credit the whole of `active`.
#[test]
fn the_metal_budget_credits_only_the_cached_generator_and_mlx_cache() {
    assert_eq!(
        unified_budget(100, 30, 5, 20),
        metal_with_resident(100, 35, 25)
    );
    // A generator larger than MLX's live bytes cannot free more than MLX holds.
    assert_eq!(
        unified_budget(100, 10, 0, 50),
        metal_with_resident(100, 10, 10)
    );
    assert_eq!(
        unified_budget(100, 30, 5, 0),
        metal_with_resident(100, 35, 5)
    );
}

/// `check` evicts the cached generator before admitting a render that needs its memory.
/// Mutation: drop the evict call in `check`'s `AdmitAfterEvict` arm.
#[test]
fn check_evicts_the_cached_generator_when_the_render_needs_its_memory() {
    let _serial = lease_test_serial();
    block_on(async {
        let s = shape(Yue2Tier::Q4, &default_request());
        let (_, floor) = priced(&s, Yue2Backend::Metal, Yue2Controls::production()).unified_floor();
        let resident = 6 << 30;
        let _budget = override_budget(Some(metal_with_resident(
            floor + resident - 1,
            resident,
            resident,
        )));
        let before = evictions();
        check(
            "yue2",
            &builtin_yue2_entry(),
            &default_request(),
            load(Yue2Tier::Q4),
            None,
            "0",
        )
        .await
        .expect("fits once evicted");
        assert_eq!(evictions(), before + 1);
    });
}
