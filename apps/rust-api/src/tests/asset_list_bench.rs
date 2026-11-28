//! Opt-in asset-list measurement against a COPY of a real project.
//!
//! Ignored by default and gated on two environment variables, so CI never runs it:
//!
//! * `SCENEWORKS_ASSET_LIST_BENCH_PROJECT` — a directory holding a real project's
//!   `project.json` and `project.db` (copied out of the user's data dir; never the live files).
//! * `SCENEWORKS_ASSET_LIST_BENCH_WORK` — a scratch directory the run may write into.
//!
//! The project is copied again into a fresh temp dir under the work dir, because opening it
//! migrates `project.db` in place.

use super::support::*;

const RUNS: usize = 5;

async fn measure(
    app: &axum::Router,
    uri: &str,
    encoding: Option<&str>,
) -> (StatusCode, usize, Vec<std::time::Duration>, Vec<u8>) {
    let headers: Vec<(&str, &str)> = encoding
        .map(|value| vec![("accept-encoding", value)])
        .unwrap_or_default();
    let mut timings = Vec::with_capacity(RUNS);
    let mut last = (StatusCode::OK, Vec::new());
    for _ in 0..RUNS {
        let started = std::time::Instant::now();
        let (status, _, body) = request_raw(app.clone(), "GET", uri, Body::empty(), &headers).await;
        timings.push(started.elapsed());
        last = (status, body);
    }
    timings.sort();
    (last.0, last.1.len(), timings, last.1)
}

fn report(label: &str, status: StatusCode, bytes: usize, timings: &[std::time::Duration]) {
    eprintln!(
        "ASSET-LIST-BENCH {label}: status={status} bytes={bytes} p50={:?} min={:?} max={:?}",
        timings[timings.len() / 2],
        timings[0],
        timings[timings.len() - 1],
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "opt-in measurement against a copied real project; see module docs"]
async fn asset_list_bench_on_copied_project() {
    let (Ok(project_dir), Ok(work_dir)) = (
        std::env::var("SCENEWORKS_ASSET_LIST_BENCH_PROJECT"),
        std::env::var("SCENEWORKS_ASSET_LIST_BENCH_WORK"),
    ) else {
        eprintln!("ASSET-LIST-BENCH skipped: set SCENEWORKS_ASSET_LIST_BENCH_PROJECT and _WORK");
        return;
    };
    let _env = isolate_hf_cache();
    let project_dir = std::path::PathBuf::from(project_dir);
    let temp_dir = tempfile::tempdir_in(&work_dir).expect("work temp dir creates");
    let summary: Value = serde_json::from_slice(
        &std::fs::read(project_dir.join("project.json")).expect("project.json reads"),
    )
    .expect("project.json parses");
    let project_id = summary["id"].as_str().expect("project id").to_owned();
    let target = temp_dir.path().join("data/projects/bench.sceneworks");
    std::fs::create_dir_all(&target).expect("project copy dir creates");
    for file in ["project.json", "project.db"] {
        std::fs::copy(project_dir.join(file), target.join(file)).expect("project file copies");
    }
    // The startup integrity pass prunes rows whose media is missing, and the copy carries no
    // media: stand in an empty file for each indexed one.
    {
        let connection = rusqlite::Connection::open(target.join("project.db")).expect("copy opens");
        let mut statement = connection.prepare("select file_path from assets").unwrap();
        let paths = statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        for path in paths {
            let media = target.join(path);
            std::fs::create_dir_all(media.parent().unwrap()).unwrap();
            std::fs::write(media, b"").unwrap();
        }
    }
    std::fs::write(
        temp_dir.path().join("data/recent-projects.json"),
        serde_json::to_vec(&json!([{
            "id": project_id,
            "name": summary["name"],
            "path": target.to_string_lossy(),
        }]))
        .unwrap(),
    )
    .expect("project registry writes");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let base = format!("/api/v1/projects/{project_id}/assets");
    let full = format!("{base}?includeRejected=true&includeTrashed=true");
    // First open migrates the copy; keep it out of the measured runs.
    let (status, _, body) = request_raw(app.clone(), "GET", &full, Body::empty(), &[]).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let count = serde_json::from_slice::<Vec<Value>>(&body)
        .expect("list parses")
        .len();
    eprintln!("ASSET-LIST-BENCH assets={count}");

    for (label, uri) in [
        ("full", full.clone()),
        ("summary", format!("{full}&view=summary")),
        ("changes-initial", format!("{full}&view=summary&since=")),
    ] {
        let (status, bytes, timings, body) = measure(&app, &uri, None).await;
        report(&format!("{label} identity"), status, bytes, &timings);
        let (status, bytes, timings, _) =
            measure(&app, &uri, Some("gzip, deflate, br, zstd")).await;
        report(
            &format!("{label} browser-encoding"),
            status,
            bytes,
            &timings,
        );
        if label == "changes-initial" && status == StatusCode::OK {
            let changes: Value = serde_json::from_slice(&body).expect("changes parse");
            let cursor = changes["cursor"].as_str().expect("cursor").to_owned();
            let since = format!("{full}&view=summary&since={cursor}");
            let (status, bytes, timings, _) = measure(&app, &since, None).await;
            report("changes-since-unchanged identity", status, bytes, &timings);
        }
    }
}
