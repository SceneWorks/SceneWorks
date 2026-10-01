use super::*;

const SCORE: &str = include_str!("fixtures/score.abc");

fn base(kind: Yue2JobKind) -> Yue2JobSpec {
    let mut spec: Yue2JobSpec =
        serde_json::from_value(serde_json::json!({ "kind": kind.as_str() })).unwrap();
    match kind {
        Yue2JobKind::Create | Yue2JobKind::Plan => {
            spec.lyrics = Some("[verse]\nla la".into());
        }
        Yue2JobKind::Cover => {
            spec.lyrics = Some("[verse]\nla la".into());
            spec.cover = Some(CoverSpec {
                mode: Some(CoverMode::Full),
                score: Some(SCORE.into()),
                ..Default::default()
            });
        }
        Yue2JobKind::FromPlan => spec.plan_job_id = Some("job-plan".into()),
        Yue2JobKind::Decode => spec.source_job_id = Some("job-song".into()),
        Yue2JobKind::RenderVersion => spec.version_id = Some("yue2v_0123".into()),
        Yue2JobKind::Transcribe => spec.source_audio_asset_id = Some("asset-1".into()),
    }
    spec
}

/// Set one [`FIELD_KINDS`] field to a valid value.
fn with_field(mut spec: Yue2JobSpec, field: &str) -> Yue2JobSpec {
    match field {
        "style" => spec.style = Some("pop".into()),
        "lyrics" => spec.lyrics = Some("[verse]\nla".into()),
        "seed" => spec.seed = Some(7),
        "cfgScale" => spec.cfg_scale = Some(1.5),
        "steps" => spec.steps = Some(8),
        "planning" => spec.planning = Some(Planning::Melody),
        "score" => spec.score = Some(SCORE.into()),
        "scoreSampling" => spec.score_sampling = Some(TokenSampling::default()),
        "semanticSampling" => spec.semantic_sampling = Some(TokenSampling::default()),
        "decoder" => spec.decoder = Some(Decoder::Legacy),
        "tier" => spec.tier = Some(Tier::Q8),
        "precision" => spec.precision = Some(ComputePrecision::Fp32),
        "arMode" => {
            spec.ar_mode = Some(ArMode::ExperimentalFp8);
            spec.tier = Some(Tier::Bf16);
        }
        "offloadPolicy" => spec.offload_policy = Some(OffloadPolicy::Sequential),
        "memory.acoustic" => {
            spec.memory = Some(MemoryControls {
                stage_residency: Some(true),
                ..Default::default()
            })
        }
        "memory.decode" => {
            spec.memory = Some(MemoryControls {
                tile_vae_decode: Some(true),
                decode_tile_edge: Some(64),
                ..Default::default()
            })
        }
        "planJobId" => spec.plan_job_id = Some("job-x".into()),
        "sourceJobId" => spec.source_job_id = Some("job-y".into()),
        "versionId" => spec.version_id = Some("yue2v_x".into()),
        "cover" => {
            spec.cover = Some(CoverSpec {
                mode: Some(CoverMode::Melody),
                score: Some(SCORE.into()),
                ..Default::default()
            })
        }
        "sourceAudioAssetId" => spec.source_audio_asset_id = Some("asset-2".into()),
        "transcription" => {
            spec.transcription = Some(TranscriptionSettings {
                max_seconds: Some(30.0),
                ..Default::default()
            })
        }
        "count" => spec.count = Some(3),
        other => panic!("unknown field {other}"),
    }
    spec
}

#[test]
fn every_kind_accepts_its_minimal_request() {
    for kind in Yue2JobKind::ALL {
        validate_request(&base(kind)).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
    }
}

#[test]
fn experimental_fp8_is_explicit_and_only_on_bf16_generation() {
    for kind in [
        Yue2JobKind::Create,
        Yue2JobKind::Plan,
        Yue2JobKind::FromPlan,
        Yue2JobKind::Cover,
        Yue2JobKind::RenderVersion,
    ] {
        let mut spec = base(kind);
        assert!(serde_json::to_value(&spec).unwrap().get("arMode").is_none());
        spec.ar_mode = Some(ArMode::ExperimentalFp8);
        spec.tier = Some(Tier::Bf16);
        validate_request(&spec).unwrap();
        assert_eq!(
            serde_json::to_value(&spec).unwrap()["arMode"],
            "experimentalFp8"
        );
        spec.tier = None;
        assert_eq!(validate_request(&spec).unwrap_err().field, "arMode");
        for tier in [Tier::Q8, Tier::Q4] {
            spec.tier = Some(tier);
            assert_eq!(validate_request(&spec).unwrap_err().field, "arMode");
        }
        spec.tier = Some(Tier::Bf16);
        spec.precision = Some(ComputePrecision::Fp32);
        assert_eq!(validate_request(&spec).unwrap_err().field, "arMode");
    }
    for kind in [Yue2JobKind::Decode, Yue2JobKind::Transcribe] {
        let mut spec = base(kind);
        spec.ar_mode = Some(ArMode::ExperimentalFp8);
        assert_eq!(validate_request(&spec).unwrap_err().field, "arMode");
    }
    let old: Yue2JobSpec = serde_json::from_value(serde_json::json!({
        "kind": "create", "lyrics": "la"
    }))
    .unwrap();
    assert_eq!(old.ar_mode, None);
    assert_eq!(serde_json::to_value(&old).unwrap().get("arMode"), None);
}

#[test]
fn every_field_is_accepted_exactly_by_the_kinds_that_read_it() {
    // The whole matrix, both ways. Mutation that reds this: adding a kind to a FIELD_KINDS row
    // (e.g. `seed` for fromPlan) makes the "refused" half accept it; removing one makes the
    // "accepted" half refuse it.
    for (field, kinds) in FIELD_KINDS {
        for kind in Yue2JobKind::ALL {
            let mut spec = with_field(base(kind), field);
            // A score needs planning that has one; a cover sent to another kind carries its own.
            if *field == "score" {
                spec.planning = None;
            }
            let result = validate_request(&spec);
            if kinds.contains(&kind) {
                assert!(result.is_ok(), "{field} on {kind:?}: {result:?}");
            } else {
                let err = result.expect_err(&format!("{field} on {kind:?} must be refused"));
                assert_eq!(err.code, INVALID_COMBINATION, "{field} on {kind:?}: {err}");
            }
        }
    }
}

/// Independent of [`FIELD_KINDS`] (the matrix test reads its expectations from the table under
/// test): every (kind, field) the `yue2` provider reads for that kind — its `map_request` arms and
/// the `LoadSpec` / memory controls each kind loads with. Mutation that reds this: removing
/// `K_COVER` from the `steps` row of FIELD_KINDS (a cover's ODE steps would be refused).
#[test]
fn every_field_the_engine_reads_for_a_kind_is_accepted() {
    use Yue2JobKind::*;
    let load = ["tier", "precision"];
    let render = ["decoder", "memory.decode"];
    let synthesis = [
        "steps",
        "semanticSampling",
        "offloadPolicy",
        "memory.acoustic",
    ];
    let sampled = ["style", "lyrics", "seed", "cfgScale"];
    let accepted: Vec<(Yue2JobKind, Vec<&str>)> = vec![
        (
            Create,
            [
                &sampled[..],
                &["planning", "score", "scoreSampling", "count"],
                &synthesis,
                &render,
                &load,
            ]
            .concat(),
        ),
        (
            Plan,
            [
                &sampled[..],
                &["planning", "score", "scoreSampling", "count"],
                &load,
            ]
            .concat(),
        ),
        (
            FromPlan,
            [
                &["style", "lyrics", "planJobId"][..],
                &synthesis,
                &render,
                &load,
            ]
            .concat(),
        ),
        (
            Cover,
            [
                &sampled[..],
                &["cover", "count"],
                &synthesis,
                &render,
                &load,
            ]
            .concat(),
        ),
        (
            RenderVersion,
            [&["versionId"][..], &synthesis, &render, &load].concat(),
        ),
        (Decode, [&["sourceJobId"][..], &render, &load].concat()),
        (Transcribe, vec!["sourceAudioAssetId", "transcription"]),
    ];
    for (kind, fields) in accepted {
        for field in fields {
            let mut spec = with_field(base(kind), field);
            if field == "score" {
                spec.planning = None;
            }
            if let Err(error) = validate_request(&spec) {
                panic!("{field} on {kind:?} is read by the engine but refused: {error}");
            }
        }
    }
}

#[test]
fn a_duplicate_gets_a_fresh_run_and_leaves_its_batch() {
    let mut payload = serde_json::json!({
        "yue2": {"kind": "create", "runId": "yue2run_original", "batch": {"id": "b", "index": 1, "count": 2}}
    })
    .as_object()
    .unwrap()
    .clone();
    refresh_block_for_duplicate(&mut payload);
    let run_id = payload["yue2"]["runId"].as_str().unwrap().to_owned();
    // Mutation that reds this: keeping the original run id (a duplicate would share its run dir).
    assert_ne!(run_id, "yue2run_original");
    assert!(run_id.starts_with(RUN_ID_PREFIX) && is_run_dir(&run_dir(&run_id)));
    assert!(payload["yue2"].get("batch").is_none());
    let mut plain = serde_json::json!({"model": "kokoro_82m"})
        .as_object()
        .unwrap()
        .clone();
    refresh_block_for_duplicate(&mut plain);
    assert_eq!(plain.len(), 1, "a payload without a block is untouched");
}

#[test]
fn each_kind_names_its_missing_input() {
    type Mutate = fn(&mut Yue2JobSpec);
    let cases: [(Yue2JobKind, Mutate, &str); 6] = [
        (Yue2JobKind::Create, |s| s.lyrics = None, "lyrics"),
        (
            Yue2JobKind::Plan,
            |s| s.lyrics = Some("  ".into()),
            "lyrics",
        ),
        (Yue2JobKind::FromPlan, |s| s.plan_job_id = None, "planJobId"),
        (
            Yue2JobKind::Decode,
            |s| s.source_job_id = None,
            "sourceJobId",
        ),
        (
            Yue2JobKind::RenderVersion,
            |s| s.version_id = None,
            "versionId",
        ),
        (
            Yue2JobKind::Cover,
            |s| s.cover.as_mut().unwrap().score = None,
            "cover.score",
        ),
    ];
    for (kind, mutate, field) in cases {
        let mut spec = base(kind);
        mutate(&mut spec);
        let err = validate_request(&spec).unwrap_err();
        assert_eq!(
            (err.code, err.field.as_str()),
            (MISSING_FIELD, field),
            "{kind:?}"
        );
    }
}

#[test]
fn protocol_limits_are_refused_with_the_field_named() {
    let mut cases: Vec<(Yue2JobSpec, &str)> = Vec::new();
    let mut s = base(Yue2JobKind::Create);
    s.cfg_scale = Some(20.5);
    cases.push((s, "cfgScale"));
    let mut s = base(Yue2JobKind::Create);
    s.seed = Some(1 << 63);
    cases.push((s, "seed"));
    let mut s = base(Yue2JobKind::Create);
    s.steps = Some(0);
    cases.push((s, "steps"));
    let mut s = base(Yue2JobKind::Create);
    s.semantic_sampling = Some(TokenSampling {
        max_tokens: Some(CONTEXT_TOKENS + 1),
        ..Default::default()
    });
    cases.push((s, "semanticSampling.maxTokens"));
    let mut s = base(Yue2JobKind::Create);
    // Only the minimum is overridden: it is checked against the phase's default maximum (4096).
    s.score_sampling = Some(TokenSampling {
        min_tokens: Some(5000),
        ..Default::default()
    });
    cases.push((s, "scoreSampling.minTokens/maxTokens"));
    let mut s = base(Yue2JobKind::Create);
    s.score_sampling = Some(TokenSampling {
        top_p: Some(0.0),
        ..Default::default()
    });
    cases.push((s, "scoreSampling.topP"));
    let mut s = base(Yue2JobKind::Create);
    s.semantic_sampling = Some(TokenSampling {
        penalty_window: Some(101),
        ..Default::default()
    });
    cases.push((s, "semanticSampling.penaltyWindow"));
    let mut s = base(Yue2JobKind::Create);
    s.memory = Some(MemoryControls {
        tile_vae_decode: Some(true),
        decode_tile_edge: Some(MAX_DECODE_TILE_FRAMES + 1),
        ..Default::default()
    });
    cases.push((s, "memory.decodeTileEdge"));
    // sc-23001: the engine refuses a chunk below one query row at the full context.
    let mut s = base(Yue2JobKind::Create);
    s.memory = Some(MemoryControls {
        chunk_attention: Some(true),
        attention_chunk_size: Some(MIN_ATTENTION_CHUNK_ELEMENTS - 1),
        ..Default::default()
    });
    cases.push((s, "memory.attentionChunkSize"));
    let mut s = base(Yue2JobKind::Create);
    s.count = Some(MAX_BATCH + 1);
    cases.push((s, "count"));
    for (spec, field) in cases {
        let err = validate_request(&spec).unwrap_err();
        assert_eq!(
            (err.code, err.field.as_str()),
            (INVALID_VALUE, field),
            "{err}"
        );
    }
}

#[test]
fn dependent_combinations_are_refused() {
    let refused = |mutate: fn(&mut Yue2JobSpec), kind: Yue2JobKind, field: &str| {
        let mut spec = base(kind);
        mutate(&mut spec);
        let err = validate_request(&spec).unwrap_err();
        assert_eq!(
            (err.code, err.field.as_str()),
            (INVALID_COMBINATION, field),
            "{err}"
        );
    };
    refused(
        |s| {
            s.planning = Some(Planning::Off);
            s.score = Some(SCORE.into());
        },
        Yue2JobKind::Create,
        "score",
    );
    refused(
        |s| {
            s.score = Some(SCORE.into());
            s.score_sampling = Some(TokenSampling::default());
        },
        Yue2JobKind::Create,
        "scoreSampling",
    );
    refused(
        |s| {
            s.planning = Some(Planning::Off);
            s.score_sampling = Some(TokenSampling::default());
        },
        Yue2JobKind::Create,
        "scoreSampling",
    );
    refused(
        |s| s.planning = Some(Planning::Off),
        Yue2JobKind::Plan,
        "planning",
    );
    refused(
        |s| {
            s.memory = Some(MemoryControls {
                attention_chunk_size: Some(4096),
                ..Default::default()
            })
        },
        Yue2JobKind::Create,
        "memory.attentionChunkSize",
    );
    refused(
        |s| s.cover.as_mut().unwrap().keep = Some(CoverKeep::Vocal),
        Yue2JobKind::Cover,
        "cover.keep",
    );
    refused(
        |s| s.cover.as_mut().unwrap().version_id = Some("yue2v_1".into()),
        Yue2JobKind::Cover,
        "cover",
    );
    refused(
        |s| s.run_id = Some("yue2run_x".into()),
        Yue2JobKind::Create,
        "runId",
    );
}

#[test]
fn an_out_of_dialect_score_is_refused_as_unsupported_notation() {
    let mut spec = base(Yue2JobKind::Create);
    spec.score = Some("X:1\nK:C\nabc".into());
    let err = validate_request(&spec).unwrap_err();
    assert_eq!(err.code, "yue2_unsupported_notation", "{err}");
}

#[test]
fn execution_requires_the_server_resolved_inputs() {
    let mut spec = base(Yue2JobKind::FromPlan);
    let err = validate_for_execution(&spec).unwrap_err();
    assert_eq!(err.field, "runId");
    spec.run_id = Some("yue2run_abc".into());
    let err = validate_for_execution(&spec).unwrap_err();
    assert_eq!(
        (err.code, err.field.as_str()),
        (MISSING_FIELD, "sources.plan")
    );
    spec.sources = Some(Sources {
        plan: Some(RunSource {
            job_id: "job-plan".into(),
            run_dir: "yue2/runs/../../etc".into(),
            identity: "ab".into(),
            plan_identity: None,
        }),
        ..Default::default()
    });
    // Mutation that reds this: accepting any `runDir` lets a stored payload point the worker
    // outside the project's run directory.
    let err = validate_for_execution(&spec).unwrap_err();
    assert_eq!(err.field, "sources.runDir");
    spec.sources
        .as_mut()
        .unwrap()
        .plan
        .as_mut()
        .unwrap()
        .run_dir = run_dir("yue2run_plan");
    validate_for_execution(&spec).unwrap();
}

/// AT2 says "reviewed": a cover from a recording is a transcribe job, a reviewed score version and
/// a cover of it. A cover naming the recording directly would skip the review, so it is refused
/// with its own code — alone or beside a score. Mutation that reds this: dropping the
/// `source_audio_asset_id` check in `validate_common` (the cover with a score then validates).
#[test]
fn a_cover_naming_a_recording_directly_requires_the_reviewed_transcription_step() {
    for keep_score in [false, true] {
        let mut cover = base(Yue2JobKind::Cover);
        let c = cover.cover.as_mut().unwrap();
        if !keep_score {
            c.score = None;
        }
        c.source_audio_asset_id = Some("asset-9".into());
        let err = validate_request(&cover).unwrap_err();
        assert_eq!(
            (err.code, err.field.as_str()),
            (TRANSCRIPTION_REVIEW_REQUIRED, "cover.sourceAudioAssetId"),
            "{err}"
        );
        assert!(err.message.contains("transcribe"), "{err}");
        assert!(err.message.contains("cover.versionId"), "{err}");
    }
}

/// The transcription settings follow the engine's window-plan rule over the RESOLVED values:
/// `0 <= lookahead <= overlap < 300`, defaults 200 / 100, a finite positive crop. Mutations that
/// red this: comparing the raw (unresolved) values (an overlap of 50 with the default look-ahead
/// of 100 then passes), or allowing overlap == 300.
#[test]
fn transcription_settings_are_held_to_the_engines_window_plan() {
    let with = |settings: TranscriptionSettings| {
        let mut spec = base(Yue2JobKind::Transcribe);
        spec.transcription = Some(settings);
        validate_request(&spec)
    };
    for ok in [
        TranscriptionSettings::default(),
        TranscriptionSettings {
            max_seconds: Some(12.5),
            overlap_seconds: Some(150.0),
            lookahead_seconds: Some(0.0),
        },
        TranscriptionSettings {
            overlap_seconds: Some(299.0),
            lookahead_seconds: Some(299.0),
            ..Default::default()
        },
    ] {
        with(ok).unwrap_or_else(|e| panic!("{ok:?}: {e}"));
    }
    for (bad, field) in [
        (
            TranscriptionSettings {
                max_seconds: Some(0.0),
                ..Default::default()
            },
            "transcription.maxSeconds",
        ),
        (
            TranscriptionSettings {
                max_seconds: Some(f64::NAN),
                ..Default::default()
            },
            "transcription.maxSeconds",
        ),
        (
            TranscriptionSettings {
                overlap_seconds: Some(TRANSCRIPTION_WINDOW_SECONDS),
                ..Default::default()
            },
            "transcription.overlapSeconds",
        ),
        (
            // The default look-ahead (100) exceeds this overlap.
            TranscriptionSettings {
                overlap_seconds: Some(50.0),
                ..Default::default()
            },
            "transcription.lookaheadSeconds",
        ),
        (
            TranscriptionSettings {
                lookahead_seconds: Some(-1.0),
                ..Default::default()
            },
            "transcription.lookaheadSeconds",
        ),
    ] {
        let err = with(bad).unwrap_err();
        assert_eq!(
            (err.code, err.field.as_str()),
            (INVALID_VALUE, field),
            "{bad:?}: {err}"
        );
    }
}

/// A queued transcribe job must carry the server-resolved recording, and that recording must be
/// the job's own asset with a SHA-256 the worker can check the file against. Mutation that reds
/// this: dropping the `K_TRANSCRIBE => need(...)` arm (a transcribe job with no recording runs).
#[test]
fn a_transcribe_job_executes_only_with_its_resolved_recording() {
    let mut spec = base(Yue2JobKind::Transcribe);
    spec.run_id = Some("yue2run_t1".into());
    let err = validate_for_execution(&spec).unwrap_err();
    assert_eq!(
        (err.code, err.field.as_str()),
        (MISSING_FIELD, "sources.recording")
    );
    let recording = |asset: &str, sha: &str| Sources {
        recording: Some(RecordingSource {
            asset_id: asset.into(),
            sha256: sha.into(),
            name: Some("take.wav".into()),
        }),
        ..Default::default()
    };
    spec.sources = Some(recording("asset-other", &"a".repeat(64)));
    assert_eq!(
        validate_for_execution(&spec).unwrap_err().field,
        "sources.recording"
    );
    spec.sources = Some(recording("asset-1", "not-a-digest"));
    assert_eq!(
        validate_for_execution(&spec).unwrap_err().field,
        "sources.recording.sha256"
    );
    spec.sources = Some(recording("asset-1", &"a".repeat(64)));
    validate_for_execution(&spec).unwrap();
    assert_eq!(
        transcription_id("yue2run_t1").as_deref(),
        Some("yue2t_t1"),
        "the record id is derived from the run id"
    );
    assert!(is_transcription_dir(&transcription_dir("yue2run_t1")));
    assert!(!is_transcription_dir(
        "yue2/transcriptions/../runs/yue2run_t1"
    ));
}

/// Independent of [`FIELD_KINDS`] (the matrix test above reads its expectations from the table
/// itself, so it cannot catch a wrong row): the combinations the `yue2` provider refuses
/// (`candle_audio_yue2::provider::map_request`'s `refuse_stray` lists) and the ones a kind would
/// silently ignore. Mutation that reds this: adding `fromPlan` to the `seed` row of FIELD_KINDS.
#[test]
fn the_engine_contracts_refusals_are_refused_here_too() {
    for (kind, field) in [
        (Yue2JobKind::FromPlan, "seed"),
        (Yue2JobKind::FromPlan, "cfgScale"),
        (Yue2JobKind::FromPlan, "planning"),
        (Yue2JobKind::FromPlan, "score"),
        (Yue2JobKind::FromPlan, "scoreSampling"),
        (Yue2JobKind::FromPlan, "cover"),
        (Yue2JobKind::Decode, "style"),
        (Yue2JobKind::Decode, "lyrics"),
        (Yue2JobKind::Decode, "seed"),
        (Yue2JobKind::Decode, "cfgScale"),
        (Yue2JobKind::Decode, "steps"),
        (Yue2JobKind::Decode, "planning"),
        (Yue2JobKind::Decode, "scoreSampling"),
        (Yue2JobKind::Decode, "semanticSampling"),
        (Yue2JobKind::Decode, "memory.acoustic"),
        (Yue2JobKind::Cover, "planning"),
        (Yue2JobKind::Cover, "score"),
        (Yue2JobKind::Plan, "steps"),
        (Yue2JobKind::Plan, "semanticSampling"),
        (Yue2JobKind::Plan, "decoder"),
        (Yue2JobKind::Plan, "memory.decode"),
        (Yue2JobKind::RenderVersion, "style"),
        (Yue2JobKind::RenderVersion, "seed"),
        (Yue2JobKind::RenderVersion, "score"),
        (Yue2JobKind::RenderVersion, "count"),
        (Yue2JobKind::FromPlan, "count"),
        (Yue2JobKind::Decode, "count"),
    ] {
        let mut spec = with_field(base(kind), field);
        if field == "score" {
            spec.planning = None;
        }
        let err =
            validate_request(&spec).expect_err(&format!("{field} on {kind:?} must be refused"));
        assert_eq!(err.code, INVALID_COMBINATION, "{field} on {kind:?}: {err}");
    }
}
