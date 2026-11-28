//! `GET /api/v1/projects/:project_id/assets`: the `view` and `since` contract.

use super::support::*;

async fn upload(app: &axum::Router, project_id: &str, name: &str) -> String {
    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        name,
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{asset}");
    asset["id"].as_str().expect("asset id").to_owned()
}

fn ids(values: &Value) -> std::collections::BTreeSet<String> {
    values
        .as_array()
        .expect("array")
        .iter()
        .map(|asset| asset["id"].as_str().unwrap().to_owned())
        .collect()
}

#[tokio::test]
async fn asset_list_view_and_since_contract() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = create_project(app.clone(), &"Listing").await;
    let project_id = project["id"].as_str().unwrap().to_owned();
    let base = format!("/api/v1/projects/{project_id}/assets");
    let first = upload(&app, &project_id, "first.png").await;
    let second = upload(&app, &project_id, "second.png").await;

    // Default: the historical bare array of full envelopes, unmarked.
    let (status, full) = request(app.clone(), "GET", &base, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ids(&full), [first.clone(), second.clone()].into());
    assert!(full[0].get("listView").is_none());
    let (status, explicit_full) = request(
        app.clone(),
        "GET",
        &format!("{base}?view=full"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(explicit_full, full);

    // view=summary: still a bare array, every item marked.
    let (status, summary) = request(
        app.clone(),
        "GET",
        &format!("{base}?view=summary"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ids(&summary), ids(&full));
    assert!(summary
        .as_array()
        .unwrap()
        .iter()
        .all(|asset| asset["listView"] == "summary"));
    // The detail route is the full envelope.
    let (status, detail) =
        request(app.clone(), "GET", &format!("{base}/{first}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(detail.get("listView").is_none());

    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("{base}?view=compact"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("{base}?since=not-a-cursor"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // since= (empty): the envelope with everything and a first cursor.
    let (status, initial) = request(
        app.clone(),
        "GET",
        &format!("{base}?view=summary&since="),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(initial["reset"], true);
    assert_eq!(ids(&initial["assets"]), ids(&full));
    assert_eq!(initial["removedIds"], json!([]));
    let cursor = initial["cursor"].as_str().expect("cursor").to_owned();

    // Only the new asset and the changed one come back.
    let third = upload(&app, &project_id, "third.png").await;
    let (status, _) = request(
        app.clone(),
        "PATCH",
        &format!("{base}/{second}/status"),
        json!({ "trashed": true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, delta) = request(
        app.clone(),
        "GET",
        &format!("{base}?view=summary&since={cursor}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(delta["reset"], false);
    assert_eq!(ids(&delta["assets"]), [third.clone()].into());
    assert_eq!(delta["removedIds"], json!([second.clone()]));
    assert_eq!(delta["assets"][0]["listView"], "summary");

    // A client that includes hidden assets sees the trashed one as a change instead.
    let (_, hidden) = request(
        app.clone(),
        "GET",
        &format!("{base}?includeTrashed=true&includeRejected=true&since={cursor}"),
        Value::Null,
    )
    .await;
    assert_eq!(
        ids(&hidden["assets"]),
        [second.clone(), third.clone()].into()
    );
    assert_eq!(hidden["removedIds"], json!([]));
    assert!(
        hidden["assets"][0].get("listView").is_none(),
        "full view by default"
    );

    // Unchanged since the latest cursor: an empty delta.
    let latest = delta["cursor"].as_str().unwrap();
    let (_, unchanged) = request(
        app.clone(),
        "GET",
        &format!("{base}?view=summary&since={latest}"),
        Value::Null,
    )
    .await;
    assert_eq!(unchanged["reset"], false);
    assert_eq!(unchanged["assets"], json!([]));
    assert_eq!(unchanged["removedIds"], json!([]));
    assert_eq!(unchanged["cursor"], delta["cursor"]);

    // A changed asset outside the character filter is reported for removal, not returned.
    let (_, character_scoped) = request(
        app.clone(),
        "GET",
        &format!("{base}?characterId=character_none&since={cursor}"),
        Value::Null,
    )
    .await;
    assert_eq!(character_scoped["assets"], json!([]));
    let removed: std::collections::BTreeSet<String> = character_scoped["removedIds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(removed, [second, third].into());
}
