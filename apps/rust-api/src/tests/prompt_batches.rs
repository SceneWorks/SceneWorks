//! rust-api prompt_batches tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[tokio::test]
async fn prompt_batch_crud_routes_persist_global_and_project_batches() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // Create a global batch with templated prompts + multi-valued variables.
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/prompt-batches",
        json!({
            "name": "Character Turnaround",
            "prompts": [
                "{{name}} with {{hair}} hair, front view",
                "{{name}} with {{hair}} hair, profile"
            ],
            "variables": [
                { "key": "name", "values": ["Alice"] },
                { "key": "hair", "values": ["red", "blue"] }
            ],
            "lastValues": { "name": ["Alice"], "hair": ["red", "blue"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["id"], "character_turnaround");
    assert_eq!(created["scope"], "global");
    assert_eq!(
        created["prompts"][0],
        "{{name}} with {{hair}} hair, front view"
    );
    assert_eq!(created["variables"][1]["values"][1], "blue");
    assert!(created["createdAt"].is_string());
    assert!(created["updatedAt"].is_string());

    let (status, list) = request(app.clone(), "GET", "/api/v1/prompt-batches", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list.as_array().expect("list array").len(), 1);
    assert_eq!(list[0]["id"], "character_turnaround");

    let (status, fetched) = request(
        app.clone(),
        "GET",
        "/api/v1/prompt-batches/character_turnaround",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["name"], "Character Turnaround");

    // Patch replaces prompts + variables and stamps updatedAt.
    let (status, updated) = request(
        app.clone(),
        "PATCH",
        "/api/v1/prompt-batches/character_turnaround",
        json!({
            "prompts": ["{{name}} smiling"],
            "variables": [{ "key": "name", "values": ["Bob"] }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["prompts"].as_array().expect("prompts").len(), 1);
    assert_eq!(updated["variables"][0]["values"][0], "Bob");

    // Duplicate carries the current (patched) state under a copied id/name.
    let (status, duplicated) = request(
        app.clone(),
        "POST",
        "/api/v1/prompt-batches/character_turnaround/duplicate",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(duplicated["id"], "character_turnaround_copy");
    assert_eq!(duplicated["name"], "Character Turnaround Copy");
    assert_eq!(duplicated["prompts"][0], "{{name}} smiling");

    // Non-string variable values are rejected.
    let (status, bad) = request(
        app.clone(),
        "POST",
        "/api/v1/prompt-batches",
        json!({ "name": "Bad", "variables": [{ "key": "x", "values": [1, 2] }] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad["detail"],
        "Prompt batch variable values must be an array of strings"
    );

    // Read-only-ish scopes are rejected on the query.
    let (status, _) = request(
        app.clone(),
        "GET",
        "/api/v1/prompt-batches?scope=builtin",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Project-scoped batch lives in the project's own manifest.
    let (status, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Batch Project" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, project_batch) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/prompt-batches?scope=project&projectId={project_id}"),
        json!({
            "name": "Project Batch",
            "prompts": ["{{subject}} portrait"],
            "variables": [{ "key": "subject", "values": ["cat"] }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(project_batch["scope"], "project");

    // With the project in context, both global and project batches list together.
    let (status, both) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/prompt-batches?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ids: Vec<&str> = both
        .as_array()
        .expect("both array")
        .iter()
        .filter_map(|batch| batch["id"].as_str())
        .collect();
    assert!(ids.contains(&"character_turnaround"));
    assert!(ids.contains(&"project_batch"));

    // Delete soft-archives: hidden from the default list, duplicate survives.
    let (status, archived) = request(
        app.clone(),
        "DELETE",
        "/api/v1/prompt-batches/character_turnaround",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(archived["archived"], true);

    let (status, after) = request(app.clone(), "GET", "/api/v1/prompt-batches", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let remaining: Vec<&str> = after
        .as_array()
        .expect("after array")
        .iter()
        .filter_map(|batch| batch["id"].as_str())
        .collect();
    assert!(!remaining.contains(&"character_turnaround"));
    assert!(remaining.contains(&"character_turnaround_copy"));
}
