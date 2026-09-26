//! sc-22997: YuE2 score inspection, bounded edits, renders and persisted A/B listening
//! comparisons over the real HTTP routes, and the same flow through a real MCP client.
use super::support::*;

const SCORE: &str =
    include_str!("../../../../crates/sceneworks-core/src/yue2_score/fixtures/score.abc");
const SCORE_JAZZ: &str =
    include_str!("../../../../crates/sceneworks-core/src/yue2_score/fixtures/score-jazz.abc");

fn song_request() -> Value {
    json!({
        "style": "English, warm piano pop, 88 BPM",
        "lyrics": "[Verse]\nNeon fades along the lane",
        "cot": "full",
        "seed": 831001
    })
}

fn user() -> Value {
    json!({ "actor": "user", "channel": "ui" })
}

fn seed_audio_asset(
    store: &sceneworks_core::project_store::ProjectStore,
    project_id: &str,
    asset_id: &str,
) {
    let project_path = std::path::PathBuf::from(store.get_project(project_id).unwrap().path);
    let relative = format!("assets/audios/set_{asset_id}/{asset_id}.wav");
    let media = project_path.join(&relative);
    std::fs::create_dir_all(media.parent().unwrap()).unwrap();
    std::fs::write(&media, b"RIFF").unwrap();
    store
        .persist_generated_asset(
            project_id,
            "job_yue2",
            &format!("set_{asset_id}"),
            &json!({
                "type": "audio",
                "assetId": asset_id,
                "mediaPath": relative,
                "mimeType": "audio/wav",
                "displayName": asset_id,
                "createdAt": "2026-09-26T00:00:00Z",
                "mode": "text_to_music",
                "model": "yue2",
                "adapter": "yue2",
                "prompt": "x",
            }),
        )
        .unwrap();
}

fn render_body(score_sha256: &str, audio: &str, semantic_truncated: bool) -> Value {
    json!({
        "status": "completed",
        "scoreSha256": score_sha256,
        "truncated": { "abc": false, "semantic": semantic_truncated },
        "model": { "id": "m-a-p/YuE2-3B", "revision": "1a96eca688d6ae5d7f0feb88573fec89920fcd19" },
        "decoder": { "id": "m-a-p/YuE2-Vae", "revision": "95535e72a97bc0f09b8ada125d26b4009428c0e8" },
        "jobId": "job_yue2",
        "audioAssetId": audio,
        "effectiveSettings": { "seed": 831001, "odeSteps": 32 },
        "provenance": { "actor": "user", "channel": "worker" }
    })
}

#[tokio::test]
async fn yue2_score_routes_round_trip_edit_render_and_compare() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let (app, state) = create_app_with_state(settings).expect("app creates");
    let project = state.project_store.create_project("YuE2 scores").unwrap();
    seed_audio_asset(&state.project_store, &project.id, "audio_source");
    seed_audio_asset(&state.project_store, &project.id, "audio_jazz");
    let base = format!("/api/v1/projects/{}/yue2", project.id);

    // Stateless inspection: exact events, and a typed rejection for unsupported notation.
    let (status, inspection) = request(
        app.clone(),
        "POST",
        "/api/v1/yue2/score/inspect",
        json!({ "abc": SCORE }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{inspection}");
    assert_eq!(inspection["voices"]["Vocal"]["soundingNotes"], 56);
    assert_eq!(inspection["bpm"], 88);
    let tuplet = SCORE.replacen("\"C\"E2G2A2", "\"C\"(3E2G2A2", 1);
    let (status, rejected) = request(
        app.clone(),
        "POST",
        "/api/v1/yue2/score/inspect",
        json!({ "abc": tuplet }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{rejected}");
    assert_eq!(rejected["code"], "yue2_unsupported_notation");
    assert!(rejected["detail"]
        .as_str()
        .unwrap()
        .contains("not necessarily invalid"));

    // Root version.
    let (status, root) = request(
        app.clone(),
        "POST",
        &format!("{base}/score-versions"),
        json!({ "abc": SCORE, "request": song_request(), "origin": "plan", "provenance": user() }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{root}");
    let root_id = root["id"].as_str().unwrap().to_owned();
    assert_eq!(root["rootVersionId"], root["id"]);
    assert!(root["renderNotice"]
        .as_str()
        .unwrap()
        .contains("regenerates the whole recording"));

    // The released harmony edit, dry run first.
    let jazz_edit = |dry_run: bool| {
        json!({
            "operation": { "op": "replace_score", "abc": SCORE_JAZZ, "allow": { "harmony": true } },
            "brief": "Reharmonize with seventh chords; keep every note, duration, bar, section and tempo.",
            "provenance": user(),
            "dryRun": dry_run,
        })
    };
    let edits = format!("{base}/score-versions/{root_id}/edits");
    let (status, preview) = request(app.clone(), "POST", &edits, jazz_edit(true)).await;
    assert_eq!(status, StatusCode::OK, "{preview}");
    assert_eq!(preview["dryRun"], true);
    let (status, listed) = request(
        app.clone(),
        "GET",
        &format!("{base}/score-versions"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        listed["items"].as_array().unwrap().len(),
        1,
        "dry run stores nothing"
    );

    let (status, edited) = request(app.clone(), "POST", &edits, jazz_edit(false)).await;
    assert_eq!(status, StatusCode::CREATED, "{edited}");
    let child = &edited["version"];
    let child_id = child["id"].as_str().unwrap().to_owned();
    assert_eq!(child["parentVersionId"], root_id);
    assert_eq!(child["edit"]["invariants"]["match"], true);
    assert_eq!(child["edit"]["contract"]["harmony"], true);
    assert!(edited["renderNotice"]
        .as_str()
        .unwrap()
        .contains("does not edit or preserve"));

    // A "harmony-only" edit that moves one melody note is refused with the invariant report.
    let broken = SCORE_JAZZ.replacen("\"Cmaj7\"E2", "\"Cmaj7\"F2", 1);
    let (status, refused) = request(
        app.clone(),
        "POST",
        &edits,
        json!({
            "operation": { "op": "replace_score", "abc": broken, "allow": { "harmony": true } },
            "brief": "Reharmonize only.",
            "provenance": user(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{refused}");
    assert_eq!(refused["code"], "yue2_invariant_violation");
    assert_eq!(refused["context"]["match"], false);
    assert!(refused["context"]["violations"][0]
        .as_str()
        .unwrap()
        .starts_with("notes:Vocal"));

    // Arbitrary operations are not accepted.
    let (status, _) = request(
        app.clone(),
        "POST",
        &edits,
        json!({ "operation": { "op": "raw_abc", "abc": SCORE }, "brief": "x", "provenance": user() }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    // Structured tempo and form edits change only their declared constraints.
    let (status, tempo) = request(
        app.clone(),
        "POST",
        &format!("{base}/score-versions/{child_id}/edits"),
        json!({ "operation": { "op": "set_tempo", "bpm": 96 }, "brief": "Faster.", "provenance": user() }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{tempo}");
    assert_eq!(tempo["version"]["rootVersionId"], root_id);
    assert_eq!(tempo["version"]["score"]["summary"]["bpm"], 96);

    // Renders of both versions, then a persisted A/B comparison.
    let root_sha = root["score"]["sha256"].as_str().unwrap();
    let child_sha = child["score"]["sha256"].as_str().unwrap();
    let (status, mismatch) = request(
        app.clone(),
        "POST",
        &format!("{base}/score-versions/{child_id}/renders"),
        render_body(root_sha, "audio_jazz", false),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{mismatch}");
    let (status, render_a) = request(
        app.clone(),
        "POST",
        &format!("{base}/score-versions/{root_id}/renders"),
        render_body(root_sha, "audio_source", false),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{render_a}");
    let (status, render_b) = request(
        app.clone(),
        "POST",
        &format!("{base}/score-versions/{child_id}/renders"),
        render_body(child_sha, "audio_jazz", true),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{render_b}");
    assert_eq!(render_b["editBrief"], child["edit"]["brief"]);
    assert_eq!(render_b["request"], child["request"]);
    assert_eq!(render_b["wholeRecordingRegenerated"], true);
    assert_eq!(render_b["truncated"]["semantic"], true);

    let (status, comparison) = request(
        app.clone(),
        "POST",
        &format!("{base}/comparisons"),
        json!({
            "versionA": root_id,
            "versionB": child_id,
            "renderA": render_a["id"],
            "renderB": render_b["id"],
            "notes": "Sevenths audible in the chorus.",
            "provenance": user(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{comparison}");
    assert_eq!(comparison["lineage"], "b_derives_from_a");
    assert_eq!(
        comparison["symbolicDifferences"],
        json!(["harmony: chord symbols differ"])
    );
    assert_eq!(comparison["b"]["render"]["audioAssetId"], "audio_jazz");
    assert!(comparison["warnings"][0]
        .as_str()
        .unwrap()
        .contains("TRUNCATED"));
    let comparison_id = comparison["id"].as_str().unwrap();
    let (status, stored) = request(
        app.clone(),
        "GET",
        &format!("{base}/comparisons/{comparison_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        stored, comparison,
        "the comparison is persisted as returned"
    );

    let (status, detail) = request(
        app.clone(),
        "GET",
        &format!("{base}/score-versions/{child_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail["renders"].as_array().unwrap().len(), 1);
    let (status, source) = request(
        app.clone(),
        "GET",
        &format!("{base}/score-versions/{root_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        source["version"], root,
        "the source version is unchanged by its edits"
    );

    let (status, _) = request(
        app,
        "GET",
        &format!("{base}/score-versions/yue2v_missing"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn mcp_agent_round_trips_yue2_score_tools() {
    use rmcp::model::CallToolRequestParams;
    use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
    use rmcp::transport::StreamableHttpClientTransport;
    use rmcp::ServiceExt;

    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral listener");
    let addr = listener.local_addr().expect("listener addr");
    let mut settings = test_settings(&temp_dir);
    settings.trust_loopback = true;
    settings.mcp_api_url = format!("http://{addr}");
    let (app, state) = create_app_with_state(settings).expect("app creates");
    let project = state.project_store.create_project("YuE2 agent").unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await;
    });
    let client = rmcp::model::ClientInfo::default()
        .serve(StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(format!("http://{addr}/mcp")),
        ))
        .await
        .expect("MCP client initializes");

    let tools = client.list_tools(None).await.expect("tools/list");
    let names: Vec<&str> = tools.tools.iter().map(|tool| tool.name.as_ref()).collect();
    for expected in [
        "yue2_inspect_score",
        "yue2_list_score_versions",
        "yue2_get_score_version",
        "yue2_create_score_version",
        "yue2_edit_score",
        "yue2_compare_versions",
    ] {
        assert!(names.contains(&expected), "missing {expected}: {names:?}");
    }
    let edit_tool = tools
        .tools
        .iter()
        .find(|tool| tool.name == "yue2_edit_score")
        .unwrap();
    assert!(
        edit_tool
            .description
            .as_deref()
            .unwrap_or("")
            .contains("regenerates the WHOLE recording"),
        "the agent is told edits regenerate the whole recording"
    );

    let call = |name: &'static str, arguments: Value| {
        CallToolRequestParams::new(name).with_arguments(arguments.as_object().unwrap().clone())
    };
    let created = client
        .call_tool(call(
            "yue2_create_score_version",
            json!({
                "projectId": project.id,
                "abc": SCORE,
                "style": "English, warm piano pop",
                "lyrics": "[Verse]\nNeon fades",
                "origin": "plan",
                "agentName": "claude",
            }),
        ))
        .await
        .expect("create call");
    assert_ne!(created.is_error, Some(true), "{created:?}");
    let root = mcp_tool_content_json(&created);
    let root_id = root["id"].as_str().unwrap().to_owned();
    assert_eq!(
        root["provenance"],
        json!({"actor": "agent", "agentName": "claude", "channel": "mcp"})
    );

    // A structured, harmony-only reharmonization through the bounded operation set.
    let edited = client
        .call_tool(call(
            "yue2_edit_score",
            json!({
                "projectId": project.id,
                "versionId": root_id,
                "operation": {
                    "op": "reharmonize",
                    "changes": [
                        { "bar": 1, "onsetQuarters": "0", "chord": "Cmaj7" },
                        { "bar": 1, "onsetQuarters": "7/2", "chord": "G7/B" }
                    ]
                },
                "brief": "Colour bar 1 with Cmaj7 and a G7/B pickup; melody fixed.",
                "agentName": "claude",
            }),
        ))
        .await
        .expect("edit call");
    assert_ne!(edited.is_error, Some(true), "{edited:?}");
    let edited = mcp_tool_content_json(&edited);
    let child_id = edited["version"]["id"].as_str().unwrap().to_owned();
    assert_eq!(edited["version"]["parentVersionId"], root_id);
    assert_eq!(edited["version"]["edit"]["invariants"]["match"], true);

    // An invariant-violating edit comes back as an isError tool result the agent can read.
    let broken = SCORE.replacen("\"C\"E2", "\"Cmaj7\"F2", 1);
    let refused = client
        .call_tool(call(
            "yue2_edit_score",
            json!({
                "projectId": project.id,
                "versionId": root_id,
                "operation": { "op": "replace_score", "abc": broken, "allow": { "harmony": true } },
                "brief": "Reharmonize only.",
            }),
        ))
        .await
        .expect("refused call is a tool result, not a protocol error");
    assert_eq!(refused.is_error, Some(true));
    let text = format!("{refused:?}");
    assert!(text.contains("yue2_invariant_violation"), "{text}");

    let inspected = client
        .call_tool(call(
            "yue2_inspect_score",
            json!({ "projectId": project.id, "versionId": child_id }),
        ))
        .await
        .expect("inspect call");
    let inspected = mcp_tool_content_json(&inspected);
    assert_eq!(inspected["voices"]["Vocal"]["soundingNotes"], 56);
    assert_eq!(inspected["voices"]["Vocal"]["chords"][1]["chord"], "G7/B");

    let compared = client
        .call_tool(call(
            "yue2_compare_versions",
            json!({ "projectId": project.id, "versionA": root_id, "versionB": child_id, "agentName": "claude" }),
        ))
        .await
        .expect("compare call");
    assert_ne!(compared.is_error, Some(true), "{compared:?}");
    let compared = mcp_tool_content_json(&compared);
    assert_eq!(compared["lineage"], "b_derives_from_a");
    assert!(compared["warnings"][0]
        .as_str()
        .unwrap()
        .contains("symbolic only"));

    let listed = client
        .call_tool(call(
            "yue2_list_score_versions",
            json!({ "projectId": project.id }),
        ))
        .await
        .expect("list call");
    let listed = mcp_tool_content_json(&listed);
    assert_eq!(listed["items"].as_array().unwrap().len(), 2);

    let _ = client.cancel().await;
}
