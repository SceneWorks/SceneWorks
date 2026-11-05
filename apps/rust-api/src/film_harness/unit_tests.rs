use super::*;

#[test]
fn generated_asset_tags_keep_legacy_literals_and_bound_long_ids() {
    assert_eq!(harness_asset_tag("role", "courier"), "role:courier");
    assert_eq!(
        harness_asset_tag("pack", "courier-workshop-refs"),
        "pack:courier-workshop-refs"
    );

    let shared = "r".repeat(63);
    let first = harness_asset_tag("role", &format!("{shared}a"));
    let second = harness_asset_tag("role", &format!("{shared}b"));
    let sound = harness_asset_tag("role", &format!("sound_{}", "s".repeat(58)));
    let pack = harness_asset_tag("pack", &format!("film_{}", "p".repeat(59)));
    for tag in [&first, &second, &sound, &pack] {
        assert!(tag.len() <= 40, "{tag}");
    }
    assert!(first.starts_with("role:h:"), "{first}");
    assert!(sound.starts_with("role:h:"), "{sound}");
    assert!(pack.starts_with("pack:h:"), "{pack}");
    assert_ne!(first, second, "distinct legal roles need distinct facets");
    assert_eq!(
        first,
        harness_asset_tag("role", &format!("{shared}a")),
        "a replay must produce the same ownership facet"
    );
}

#[test]
fn multipart_body_carries_file_and_provenance_fields() {
    let (boundary, body) =
        encode_asset_upload("plate.png", "image/png", b"\x89PNG", &json!({ "a": 1 }));
    let text = String::from_utf8_lossy(&body);
    assert!(text.starts_with(&format!("--{boundary}\r\n")));
    assert!(text.contains("name=\"file\"; filename=\"plate.png\""));
    assert!(text.contains("Content-Type: image/png"));
    assert!(text.contains("name=\"provenance\"\r\n\r\n{\"a\":1}"));
    assert!(text.ends_with(&format!("--{boundary}--\r\n")));
}

#[test]
fn a_filename_cannot_inject_multipart_headers() {
    let (_, body) = encode_asset_upload(
        "plate\r\nX-Injected: 1\r\n\r\nevil.png",
        "image/png",
        b"\x89PNG",
        &json!({}),
    );
    let text = String::from_utf8_lossy(&body);
    assert!(!text.contains("X-Injected: 1\r\n"), "{text}");
    assert!(
        text.contains("filename=\"plate__X-Injected__1____evil.png\""),
        "{text}"
    );
    assert_eq!(sanitize_multipart_filename("a\"b.png"), "a_b.png");
    assert_eq!(
        sanitize_multipart_filename("../../etc/passwd"),
        ".._.._etc_passwd"
    );
    assert_eq!(sanitize_multipart_filename("..."), "reference.png");
}

#[test]
fn the_platform_gate_is_the_routes_own_and_follows_the_api_host() {
    let mac_only: JsonObject<String, Value> =
        json!({ "id": "some_model", "type": "video", "macOnly": true })
            .as_object()
            .cloned()
            .unwrap();
    let portable: JsonObject<String, Value> = json!({ "id": "some_model", "type": "video" })
        .as_object()
        .cloned()
        .unwrap();
    let finding = platform_reachability_finding("some_model", &mac_only, "linux")
        .expect("a mac-only model is unreachable on linux");
    assert_eq!(finding.field, "model.id");
    assert!(finding.message.contains("only on macOS"), "{finding}");
    assert!(platform_reachability_finding("some_model", &mac_only, "macos").is_none());
    assert!(platform_reachability_finding("some_model", &portable, "linux").is_none());
}

#[test]
fn the_reference_payload_gate_is_the_routes_own() {
    // Ten reference roles: past the route's blanket ceiling of nine, which no per-model
    // `limits.maxReferenceAssets` can raise. Found here rather than as a 400 at enqueue.
    let roles: Vec<String> = (0..10).map(|index| format!("role_{index}")).collect();
    let plan: ProductionPlan = serde_json::from_value(json!({
        "schemaVersion": 1,
        "id": "p", "version": 1, "title": "t",
        "model": { "id": "some_model" },
        "limits": { "maxRunSeconds": 10, "maxShotSeconds": 5, "maxAttemptsPerShot": 1, "maxMemoryGb": 8 },
        "shots": [{
            "id": "SH010", "beat": "b", "framing": "f", "prompt": "p",
            "targetDurationSeconds": 5.0, "startState": "s", "endState": "e", "audio": "Room tone, no music.",
            "conditioning": { "mode": "reference_to_video", "referenceRoles": roles }
        }]
    }))
    .expect("plan parses");
    let entry: JsonObject<String, Value> = json!({ "id": "some_model", "type": "video" })
        .as_object()
        .cloned()
        .unwrap();
    // An EMPTY pack: none of these roles is declared, so none can share a file with another
    // and the shot supplies ten images — the case this gate exists for (sc-24024).
    let findings = reference_payload_findings(
        &plan,
        &empty_pack(),
        &ModelEntries::single("some_model", &entry),
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));
    assert!(findings[0].message.contains("at most 9"), "{findings:?}");

    // The shipped shape — no references at all — passes.
    let plan: ProductionPlan = serde_json::from_value(json!({
        "schemaVersion": 1,
        "id": "p", "version": 1, "title": "t",
        "model": { "id": "some_model" },
        "limits": { "maxRunSeconds": 10, "maxShotSeconds": 5, "maxAttemptsPerShot": 1, "maxMemoryGb": 8 },
        "shots": [{
            "id": "SH010", "beat": "b", "framing": "f", "prompt": "p",
            "targetDurationSeconds": 5.0, "startState": "s", "endState": "e", "audio": "Room tone, no music.",
            "conditioning": { "mode": "text_to_video" }
        }]
    }))
    .expect("plan parses");
    assert!(reference_payload_findings(
        &plan,
        &empty_pack(),
        &ModelEntries::single("some_model", &entry)
    )
    .is_empty());
}

#[test]
fn the_reference_payload_gate_preserves_declared_keyframes_without_inventing_them() {
    let mut plan: ProductionPlan = serde_json::from_value(json!({
        "schemaVersion": 1,
        "id": "p", "version": 1, "title": "t",
        "model": { "id": "some_model" },
        "limits": { "maxRunSeconds": 10, "maxShotSeconds": 5, "maxAttemptsPerShot": 1, "maxMemoryGb": 8 },
        "shots": [{
            "id": "SH010", "beat": "b", "framing": "f", "prompt": "p",
            "targetDurationSeconds": 5.0, "startState": "s", "endState": "e", "audio": "Room tone, no music.",
            "conditioning": { "mode": "image_to_video", "firstFrameRole": "plate" }
        }]
    }))
    .unwrap();
    let entry = json!({ "id": "some_model", "type": "video" })
        .as_object()
        .unwrap()
        .clone();
    let pack = empty_pack();
    let entries = ModelEntries::single("some_model", &entry);
    assert!(reference_payload_findings(&plan, &pack, &entries).is_empty());

    // A missing source must still be rejected, even when a last frame was declared. Only
    // trusted FL2VA/hybrid entries have the route's last-frame-only exception.
    plan.shots[0].conditioning.first_frame_role = None;
    plan.shots[0].conditioning.last_frame_role = Some("end_plate".into());
    let findings = reference_payload_findings(&plan, &pack, &entries);
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].message.contains("requires a source image"));
}

/// A pack declaring nothing, for the gates whose plans name roles no pack ever approved.
fn empty_pack() -> film_plan::ReferencePack {
    serde_json::from_value(json!({
        "schemaVersion": film_plan::REFERENCE_PACK_SCHEMA_VERSION,
        "id": "empty",
        "version": 1,
        "references": [],
    }))
    .expect("the empty pack parses")
}

#[test]
fn the_lane_and_platform_follow_the_api_host_not_this_process() {
    let remote = HostFacts {
        platform: Some("linux".to_owned()),
        ..HostFacts::default()
    };
    assert_eq!(remote.lane(), ModelLane::Candle);
    assert_eq!(remote.platform_or_local(), "linux");
    let mac = HostFacts {
        platform: Some("macos".to_owned()),
        ..HostFacts::default()
    };
    assert_eq!(mac.lane(), ModelLane::Mlx);
    // Only a host that reports no platform at all falls back to this process's own.
    let unknown = HostFacts::default();
    assert_eq!(unknown.lane(), ModelLane::for_current_platform());
    assert_eq!(unknown.platform_or_local(), std::env::consts::OS);
}

#[test]
fn the_memory_peak_prefers_the_metrics_route_over_the_job_snapshot() {
    let view = |pct: Option<f64>| JobView {
        status: "completed".to_owned(),
        stage: "completed".to_owned(),
        error: None,
        message: String::new(),
        peak_gpu_memory_pct: pct,
        backend: None,
        result: Value::Null,
        reference_image_resize_mode: None,
    };
    let bytes = json!({ "peakMemoryBytes": 115.2 * BYTES_PER_GB, "peakMemoryPct": 90.0 });
    let observed = memory_observation(Some(&bytes), &view(Some(10.0)), Some(128.0));
    assert_eq!(
        observed.source.as_deref(),
        Some("metrics.peakMemoryBytes"),
        "the snapshot's 10% must not win over a measured byte count"
    );
    assert!(observed.gb.is_some_and(|gb| (gb - 115.2).abs() < 1e-6));

    let pct_only = json!({ "peakMemoryPct": 50.0 });
    let observed = memory_observation(Some(&pct_only), &view(None), Some(128.0));
    assert_eq!(observed.source.as_deref(), Some("metrics.peakMemoryPct"));
    assert_eq!(observed.gb, Some(64.0));

    // Last resort: the job snapshot's field, which every shipped worker leaves null.
    let observed = memory_observation(None, &view(Some(25.0)), Some(128.0));
    assert_eq!(observed.source.as_deref(), Some("job.peakGpuMemoryPct"));
    assert_eq!(observed.gb, Some(32.0));
    assert!(memory_observation(None, &view(None), Some(128.0))
        .gb
        .is_none());
    // A zero row (what the cpu export job records on a real run) is "nothing measured".
    let zeroed = json!({ "peakMemoryBytes": 0, "peakMemoryPct": 0.0 });
    assert!(memory_observation(Some(&zeroed), &view(None), Some(128.0))
        .gb
        .is_none());
}

#[test]
fn a_completed_job_is_settled_once_its_assets_are_persisted() {
    let with_result = |result: Value| JobView {
        status: "completed".to_owned(),
        stage: "completed".to_owned(),
        error: None,
        message: String::new(),
        peak_gpu_memory_pct: None,
        backend: None,
        result,
        reference_image_resize_mode: None,
    };
    assert!(!with_result(json!({ "assetWrites": [{ "type": "video" }] })).is_settled());
    assert!(
        !with_result(json!({ "assetWrites": [{ "type": "video" }], "assets": [] })).is_settled()
    );
    let writes = json!([{ "assetId": "asset_1" }, { "assetId": "asset_2" }]);
    assert!(
        !with_result(
            json!({ "assetWrites": writes, "assets": [{"id":"asset_1"}], "assetIds":["asset_1"] })
        )
        .is_settled(),
        "an earlier published output does not settle a later pending write"
    );
    assert!(
        !with_result(json!({ "assetWrites": writes, "assetIds":["asset_1", "unrelated"] }))
            .is_settled(),
        "matching counts cannot stand in for matching identities"
    );
    assert!(with_result(
        json!({ "assetWrites": writes, "assets": [{"id":"asset_1"}, {"id":"asset_2"}] })
    )
    .is_settled());
    assert!(
        with_result(json!({ "assetWrites": writes, "assetIds":["asset_1", "asset_2"] }))
            .is_settled()
    );
    assert!(with_result(json!({ "assetIds": ["asset_1"] })).is_settled());
    assert!(with_result(json!({ "assetWrites": [] })).is_settled());
    assert!(with_result(json!({})).is_settled());
}

#[tokio::test]
async fn an_expired_run_deadline_wins_even_when_the_shot_deadline_expired_first() {
    struct RunningJobTransport;

    impl ApiTransport for RunningJobTransport {
        fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
            Box::pin(async move {
                match (request.method, request.path.as_str()) {
                    ("GET", "/api/v1/jobs/job_running") => Ok(ApiResponse {
                        status: 200,
                        body: json!({"status": "running", "message": "still rendering"}),
                    }),
                    ("POST", "/api/v1/jobs/job_running/cancel") => Ok(ApiResponse {
                        status: 200,
                        body: Value::Null,
                    }),
                    _ => Err(HarnessError::Transport(format!(
                        "unexpected synthetic request {} {}",
                        request.method, request.path
                    ))),
                }
            })
        }

        fn get_bytes(&self, path: String) -> BytesTransportFuture<'_> {
            Box::pin(async move {
                Err(HarnessError::Transport(format!(
                    "unexpected synthetic byte request {path}"
                )))
            })
        }
    }

    let transport = RunningJobTransport;
    let control = RunControl::new();
    let client = Client {
        transport: &transport,
        control: &control,
    };
    let now = Instant::now();
    let (_, stop) = client
        .wait_for_job(
            "job_running",
            PollBounds {
                // Both limits are spent. The shot expired EARLIER, so this assertion fails if
                // the poller uses chronological order instead of the run-budget precedence.
                shot_deadline: now - Duration::from_secs(2),
                run_deadline: Some(now - Duration::from_secs(1)),
                poll_interval: Duration::ZERO,
                cancel_grace: Duration::ZERO,
                settle_grace: Duration::ZERO,
            },
        )
        .await
        .expect("synthetic running job is canceled at the expired bound");
    assert_eq!(stop, PollStop::RunBudget);
}

#[test]
fn timeline_items_are_read_back_off_the_saved_document() {
    let intended = vec!["item_sh010_abcd1234".to_owned()];
    // Sound tracks sit beside the picture track since sc-22712, so the check is scoped to the
    // track the shots live on — "as many items as intended" is only true of that one.
    let saved = json!({ "tracks": [
        { "id": "track_main", "items": [{
            "id": "item_sh010_abcd1234", "assetId": "asset_1",
            "timelineStart": 0.0, "timelineEnd": 5.1667
        }] },
        { "id": "track_dialogue", "items": [{
            "id": "item_line_sh010_abcd1234", "assetId": "asset_2",
            "timelineStart": 1.0, "timelineEnd": 3.0
        }] },
    ] });
    let persisted =
        persisted_picture_items(&saved, "track_main", &intended).expect("items read back");
    assert_eq!(persisted.len(), 1, "only the picture track is counted");
    assert!(
        (persisted[0]["timelineEnd"].as_f64().unwrap() - 5.1667).abs() < 1e-9,
        "the SAVED value is what is read, not the intended one"
    );
    // A timeline the save dropped the items from cannot be recorded as if it held them.
    let empty = json!({ "tracks": [{ "id": "track_main", "items": [] }] });
    assert!(persisted_picture_items(&empty, "track_main", &intended).is_none());
    // Nor can one whose picture track grew an item the run never assembled.
    let extra = json!({ "tracks": [{ "id": "track_main", "items": [
        { "id": "item_sh010_abcd1234", "assetId": "asset_1",
          "timelineStart": 0.0, "timelineEnd": 5.1667 },
        { "id": "item_stowaway", "assetId": "asset_9",
          "timelineStart": 5.1667, "timelineEnd": 9.0 },
    ] }] });
    assert!(persisted_picture_items(&extra, "track_main", &intended).is_none());
}

#[test]
fn a_re_derived_harness_audio_item_keeps_the_editors_volume_and_fades() {
    let harness_item = |id: &str, shot: Option<&str>, volume: f64, fade_in: f64, fade_out: f64| {
        json!({
            "id": id, "trackId": "track_dialogue", "assetId": "asset_line", "type": "audio",
            "sourceIn": 0.0, "sourceOut": 2.0, "timelineStart": 0.0, "timelineEnd": 2.0,
            "volume": volume, "fadeInSeconds": fade_in, "fadeOutSeconds": fade_out,
            HARNESS_KEY: harness_block(ROLE_DIALOGUE, "run_0123456789ab", shot, 1.2),
        })
    };
    // The saved sequence: the editor turned SH020's line down and gave it fades, left
    // SH050's line alone, and placed one clip of their own on the same track.
    let saved = json!({ "tracks": [{
        "id": "track_dialogue", "name": "Dialogue", "kind": "audio", "role": "dialogue",
        "gain": 0.7, "muted": false,
        "items": [
            harness_item("item_line_sh020", Some("SH020"), 0.4, 0.25, 0.5),
            harness_item("item_line_sh050", Some("SH050"), 1.0, 0.0, 0.0),
            { "id": "item_editor", "trackId": "track_dialogue", "assetId": "asset_foreign",
              "type": "audio", "sourceIn": 0.0, "sourceOut": 1.0,
              "timelineStart": 3.0, "timelineEnd": 4.0, "volume": 0.9 },
        ],
    }] });
    let tracks = saved["tracks"].as_array().cloned().unwrap();
    // A later pass re-derives every harness item from the plan — at the plan's gain, with the
    // plan's fades — and this time SH060 has a take, so its line appears for the first time.
    let fresh = json!({
        "id": "track_dialogue", "name": "Dialogue", "kind": "audio", "role": "dialogue",
        "gain": 1.0, "muted": false,
        "items": [
            harness_item("item_line_sh020", Some("SH020"), 1.0, 0.0, 0.0),
            harness_item("item_line_sh050", Some("SH050"), 1.0, 0.0, 0.0),
            harness_item("item_line_sh060", Some("SH060"), 0.8, 0.0, 0.4),
        ],
    });
    let merged = merge_harness_audio_track(&tracks, fresh);
    let items = merged["items"].as_array().unwrap();
    let by_id = |id: &str| {
        items
            .iter()
            .find(|item| item["id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("{id} is on the merged track"))
    };
    let sh020 = by_id("item_line_sh020");
    assert_eq!(
        sh020["volume"],
        json!(0.4),
        "the editor's volume survives the pass"
    );
    assert_eq!(sh020["fadeInSeconds"], json!(0.25));
    assert_eq!(sh020["fadeOutSeconds"], json!(0.5));
    assert_eq!(by_id("item_line_sh050")["volume"], json!(1.0));
    let sh060 = by_id("item_line_sh060");
    assert_eq!(
        sh060["volume"],
        json!(0.8),
        "an item with no saved counterpart is the fresh one"
    );
    assert_eq!(sh060["fadeOutSeconds"], json!(0.4));
    assert_eq!(
        by_id("item_editor")["volume"],
        json!(0.9),
        "the editor's own clip is kept"
    );
    assert_eq!(items.len(), 4);
    assert_eq!(merged["gain"], json!(0.7), "the saved fader wins");
    // A bed carries no shot id: it matches its saved counterpart on the role alone.
    let bed = |volume: f64| {
        json!({ "id": "track_music", "kind": "audio", "role": "music", "gain": 0.2,
            "items": [{ "id": "item_music", "type": "audio", "volume": volume,
                "fadeInSeconds": 2.0, "fadeOutSeconds": 3.0,
                HARNESS_KEY: harness_block(ROLE_MUSIC, "run_0123456789ab", None, 0.0) }] })
    };
    let merged_bed = merge_harness_audio_track(&[bed(0.5)], bed(1.0));
    assert_eq!(merged_bed["items"][0]["volume"], json!(0.5));
    // With no saved track at all, the fresh one is used as is.
    assert_eq!(
        merge_harness_audio_track(&[], bed(1.0))["items"][0]["volume"],
        json!(1.0)
    );
}

#[test]
fn sequence_beds_stop_at_the_measured_source_end_without_implicit_looping() {
    let bed = SoundBed {
        role: "notification_chime".to_owned(),
        gain: 0.6,
        muted: false,
        start_seconds: 8.2,
        source_in_seconds: 0.0,
        fade_in_seconds: 0.02,
        fade_out_seconds: 0.04,
    };
    let asset = SoundAsset {
        asset_id: "asset_chime".to_owned(),
        duration_seconds: Some(0.35),
    };
    let mut timeline = json!({ "tracks": [
        { "id": PICTURE_TRACK_ID, "kind": "video", "items": [{
            "id": "picture", "sourceIn": 0.0, "sourceOut": 24.041666666666668,
            "timelineStart": 0.0, "timelineEnd": 24.041666666666668, "speed": 1.0
        }] },
        bed_track("track_sfx_0", "Sound effect", ROLE_SFX, &bed, &asset, "run_3e86b06e")
    ] });

    let duration = relayout_timeline(&mut timeline, None).expect("timeline relayout");
    assert_eq!(duration, 24.041666666666668);
    let effect = &timeline["tracks"][1]["items"][0];
    assert_eq!(effect["timelineStart"], json!(8.2));
    assert!(
        (effect["timelineEnd"].as_f64().unwrap() - 8.55).abs() < 0.000_001,
        "{effect}"
    );
    assert_eq!(effect["sourceIn"], json!(0.0));
    assert_eq!(effect["sourceOut"], json!(0.35));
    assert_eq!(effect[HARNESS_KEY]["sourceDurationSeconds"], json!(0.35));

    let long_asset = SoundAsset {
        asset_id: "asset_music".to_owned(),
        duration_seconds: Some(32.0),
    };
    let long_bed = SoundBed {
        role: "main_theme".to_owned(),
        source_in_seconds: 0.5,
        start_seconds: 1.2,
        ..bed
    };
    timeline["tracks"][1] = bed_track(
        MUSIC_TRACK_ID,
        "Music",
        ROLE_MUSIC,
        &long_bed,
        &long_asset,
        "run_3e86b06e",
    );
    relayout_timeline(&mut timeline, None).expect("long bed relayout");
    let music = &timeline["tracks"][1]["items"][0];
    assert_eq!(music["timelineStart"], json!(1.2));
    assert_eq!(music["timelineEnd"], json!(24.041666666666668));
    assert!(
        (music["sourceOut"].as_f64().unwrap() - 23.34166666666667).abs() < 0.000_001,
        "{music}"
    );
}

// `tier_maps_to_the_shared_mlx_quantize_convention` moved to `sceneworks_core::film_compile`
// with `mlx_quantize_for_tier` itself, which now builds every video job body (sc-22713).

#[test]
fn export_resolution_is_the_smallest_admitted_value_covering_the_takes() {
    assert_eq!(export_resolution_for(320), 640);
    assert_eq!(export_resolution_for(720), 720);
    assert_eq!(export_resolution_for(768), 1024);
    assert_eq!(export_resolution_for(2000), 1280);
    // The timeline ratio is the admitted one CLOSEST to the take's real geometry, and it is a
    // coercion whenever the take is not already 16:9 / 9:16 / 1:1: the fixture's 576x320 takes
    // are 9:5, so the 16:9 timeline pads them (the real smoke's export landed at 1138x640).
    assert_eq!(aspect_ratio_for(576, 320), "16:9");
    assert_eq!(reduced_aspect_ratio(576, 320), "9:5");
    assert_eq!(aspect_ratio_for(320, 576), "9:16");
    assert_eq!(reduced_aspect_ratio(320, 576), "5:9");
    assert_eq!(aspect_ratio_for(768, 768), "1:1");
    assert_eq!(reduced_aspect_ratio(768, 768), "1:1");
    // 4:3 is exactly equidistant from 1:1 and 16:9 on the log scale (4/3 is their geometric
    // mean), so the tie-break decides — and a landscape take belongs in a landscape frame.
    assert_eq!(aspect_ratio_for(1024, 768), "16:9");
    assert_eq!(aspect_ratio_for(768, 1024), "9:16");
    assert_eq!(reduced_aspect_ratio(1024, 768), "4:3");
    // Only an exact match is not a coercion.
    assert_eq!(aspect_ratio_for(1280, 720), "16:9");
    assert_eq!(reduced_aspect_ratio(1280, 720), "16:9");
    assert_eq!(reduced_aspect_ratio(1344, 768), "7:4");
}

#[test]
fn the_export_resolution_menu_is_the_routes_own() {
    assert_eq!(
        crate::TIMELINE_EXPORT_RESOLUTIONS,
        &[640, 720, 1024, 1280],
        "the harness picks from the list validate_timeline_export enforces"
    );
}

#[test]
fn fixture_plates_are_deterministic_and_decode_at_the_declared_size() {
    let first = fixture_plate_png("red_parcel", [200, 32, 32]).unwrap();
    let second = fixture_plate_png("red_parcel", [200, 32, 32]).unwrap();
    assert_eq!(first, second);
    let decoded = image::load_from_memory(&first).unwrap();
    assert_eq!((decoded.width(), decoded.height()), FIXTURE_PLATE_SIZE);
    assert_ne!(first, fixture_plate_png("courier", [64, 80, 120]).unwrap());
}

#[test]
fn an_export_lookup_excludes_every_export_the_record_ever_held() {
    let export = |id: &str, created: &str, timeline: &str| {
        json!({
            "id": id,
            "type": "timeline_export",
            "createdAt": created,
            "payload": { "timelineId": timeline }
        })
    };
    let jobs = vec![
        export("export_1", "2026-09-13T10:00:00Z", "tl_run"),
        export("export_2", "2026-09-13T11:00:00Z", "tl_run"),
        export("export_3", "2026-09-13T12:00:00Z", "tl_other"),
        json!({
            "id": "video_1",
            "type": "video_generate",
            "createdAt": "2026-09-13T11:30:00Z",
            "payload": { "timelineId": "tl_run" }
        }),
    ];
    // The FIRST re-export: export_1 is superseded, and nothing newer than the request exists.
    assert_eq!(
        newest_export_job(
            &jobs,
            "tl_run",
            &["export_1".to_owned()],
            "2026-09-13T11:30:00Z"
        ),
        None,
        "export_2 predates this request, so it is not the job this pass created"
    );
    // The SECOND re-export: both earlier exports are excluded, so nothing is adoptable — the
    // bug was adopting export_1 here and recording its pre-replacement asset as current.
    assert_eq!(
        newest_export_job(
            &jobs,
            "tl_run",
            &["export_1".to_owned(), "export_2".to_owned()],
            "2026-09-13T09:00:00Z"
        ),
        None
    );
    // Excluding only the most recent one is exactly the defect.
    assert_eq!(
        newest_export_job(
            &jobs,
            "tl_run",
            &["export_2".to_owned()],
            "2026-09-13T09:00:00Z"
        ),
        Some("export_1".to_owned())
    );
    // The job this pass really did create is adopted: newer than the request, not excluded.
    let mut with_new = jobs.clone();
    with_new.push(export("export_4", "2026-09-13T12:30:00Z", "tl_run"));
    assert_eq!(
        newest_export_job(
            &with_new,
            "tl_run",
            &["export_1".to_owned(), "export_2".to_owned()],
            "2026-09-13T12:00:00Z"
        ),
        Some("export_4".to_owned())
    );
    // A job with no readable `createdAt` is never adopted: the harness would rather run a
    // second export than record an unrelated asset as this run's delivered MP4.
    let undated = vec![json!({
        "id": "export_5",
        "type": "timeline_export",
        "payload": { "timelineId": "tl_run" }
    })];
    assert_eq!(
        newest_export_job(&undated, "tl_run", &[], "2026-09-13T09:00:00Z"),
        None
    );
}

#[test]
fn job_pages_merge_on_id_newest_first() {
    let job = |id: &str, created: &str| json!({ "id": id, "createdAt": created });
    let merged = merge_job_pages(vec![
        vec![
            job("a", "2026-09-13T10:00:00Z"),
            job("b", "2026-09-13T12:00:00Z"),
        ],
        vec![
            job("b", "2026-09-13T12:00:00Z"),
            job("c", "2026-09-13T11:00:00Z"),
        ],
        vec![json!({ "type": "video_generate" })],
    ]);
    let ids: Vec<&str> = merged
        .iter()
        .filter_map(|job| job.get("id").and_then(Value::as_str))
        .collect();
    assert_eq!(ids, vec!["b", "c", "a"], "one entry per id, newest first");
}

#[test]
fn an_attempt_with_no_job_has_spent_nothing() {
    let attempt = |job_id: Option<&str>, started_at: &str| AttemptRecord {
        attempt: 1,
        idempotency_key: "run:SH010:a1".to_owned(),
        resolved_model_id: "minimax_h3".to_owned(),
        partition_reason: String::new(),
        reference_image_short_edge: None,
        reference_image_resize_mode: None,
        loras: Vec::new(),
        effective_steps: None,
        turbo_scheduler_shift: None,
        job_id: job_id.map(str::to_owned),
        status: "dispatching".to_owned(),
        started_at: started_at.to_owned(),
        finished_at: None,
        elapsed_seconds: 0.0,
        peak_gpu_memory_pct: None,
        peak_memory_gb: None,
        peak_memory_source: None,
        error: None,
        take: None,
        rejection: None,
        human_requested: false,
    };
    let yesterday = sceneworks_core::time::format_unix_seconds(
        sceneworks_core::time::now_unix_seconds() - 86_400,
    );
    assert_eq!(
        attempt_spent_seconds(&attempt(None, &yesterday)),
        0.0,
        "an attempt recorded before its job POST landed rendered nothing, whatever its \
         startedAt says"
    );
    assert!(
        attempt_spent_seconds(&attempt(Some("job_1"), &yesterday)) > 86_000.0,
        "an attempt that HAS a job has been in flight since it started"
    );
}

#[test]
fn installed_state_reads_the_requested_tier_variant_first() {
    let entry: JsonObject<String, Value> = json!({
        "installState": "installed",
        "variants": [
            { "variant": "q4", "installed": true },
            { "variant": "q8", "installed": false }
        ]
    })
    .as_object()
    .cloned()
    .unwrap();
    assert!(model_tier_installed(&entry, Some("q4")));
    assert!(!model_tier_installed(&entry, Some("q8")));
    assert!(model_tier_installed(&entry, None));
    assert!(
        model_tier_installed(&entry, Some("bf16")),
        "unlisted tier falls back to the model state"
    );
}
