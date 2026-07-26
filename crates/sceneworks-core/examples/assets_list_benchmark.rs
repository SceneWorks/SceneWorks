use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sceneworks_core::project_store::{AssetListFilesystemOperations, AssetScope, ProjectStore};
use serde_json::json;

fn list_once(
    store: &ProjectStore,
    project_id: &str,
) -> (usize, Duration, AssetListFilesystemOperations) {
    let started = Instant::now();
    let (assets, operations) = store
        .list_assets_with_filesystem_operations(project_id, true, true, AssetScope::All)
        .expect("asset list succeeds");
    (assets.len(), started.elapsed(), operations)
}

fn report(
    label: &str,
    assets: usize,
    elapsed: Duration,
    operations: AssetListFilesystemOperations,
) {
    println!(
        "{label}: assets={assets} elapsed_ms={:.3} fs_total={} registry_opens={} \
         registry_metadata_reads={} registry_content_reads={} path_stats={} directory_scans={} \
         sidecar_reads={} generation_set_reads={} timeline_reads={} character_reads={} \
         poster_stats={} poster_reads={} index_marker_reads={} index_marker_writes={} index_marker_removes={} \
         directory_create_calls={} db_opens={}",
        elapsed.as_secs_f64() * 1_000.0,
        operations.total(),
        operations.registry_opens,
        operations.registry_metadata_reads,
        operations.registry_content_reads,
        operations.path_stats,
        operations.directory_scans,
        operations.sidecar_reads,
        operations.generation_set_reads,
        operations.timeline_reads,
        operations.character_reads,
        operations.poster_stats,
        operations.poster_reads,
        operations.index_marker_reads,
        operations.index_marker_writes,
        operations.index_marker_removes,
        operations.directory_create_calls,
        operations.db_opens,
    );
}

fn run(store: ProjectStore, project_id: &str, iterations: usize) {
    // This is the first call through a new ProjectStore and therefore includes
    // registry-cache population. It does NOT evict the OS filesystem cache.
    let (assets, first_elapsed, first_operations) = list_once(&store, project_id);
    report("first-call", assets, first_elapsed, first_operations);

    let mut warm_elapsed = Duration::ZERO;
    let mut warm_operations = AssetListFilesystemOperations::default();
    for _ in 0..iterations {
        let (warm_assets, elapsed, operations) = list_once(&store, project_id);
        assert_eq!(warm_assets, assets, "asset count changed during benchmark");
        warm_elapsed += elapsed;
        warm_operations.registry_opens += operations.registry_opens;
        warm_operations.registry_metadata_reads += operations.registry_metadata_reads;
        warm_operations.registry_content_reads += operations.registry_content_reads;
        warm_operations.path_stats += operations.path_stats;
        warm_operations.directory_scans += operations.directory_scans;
        warm_operations.sidecar_reads += operations.sidecar_reads;
        warm_operations.generation_set_reads += operations.generation_set_reads;
        warm_operations.timeline_reads += operations.timeline_reads;
        warm_operations.character_reads += operations.character_reads;
        warm_operations.poster_stats += operations.poster_stats;
        warm_operations.poster_reads += operations.poster_reads;
        warm_operations.index_marker_reads += operations.index_marker_reads;
        warm_operations.index_marker_writes += operations.index_marker_writes;
        warm_operations.index_marker_removes += operations.index_marker_removes;
        warm_operations.directory_create_calls += operations.directory_create_calls;
        warm_operations.db_opens += operations.db_opens;
    }
    let divisor = iterations as u32;
    report(
        &format!("steady-state-average({iterations})"),
        assets,
        warm_elapsed / divisor,
        AssetListFilesystemOperations {
            registry_opens: warm_operations.registry_opens / iterations as u64,
            registry_metadata_reads: warm_operations.registry_metadata_reads / iterations as u64,
            registry_content_reads: warm_operations.registry_content_reads / iterations as u64,
            path_stats: warm_operations.path_stats / iterations as u64,
            directory_scans: warm_operations.directory_scans / iterations as u64,
            sidecar_reads: warm_operations.sidecar_reads / iterations as u64,
            generation_set_reads: warm_operations.generation_set_reads / iterations as u64,
            timeline_reads: warm_operations.timeline_reads / iterations as u64,
            character_reads: warm_operations.character_reads / iterations as u64,
            poster_stats: warm_operations.poster_stats / iterations as u64,
            poster_reads: warm_operations.poster_reads / iterations as u64,
            index_marker_reads: warm_operations.index_marker_reads / iterations as u64,
            index_marker_writes: warm_operations.index_marker_writes / iterations as u64,
            index_marker_removes: warm_operations.index_marker_removes / iterations as u64,
            directory_create_calls: warm_operations.directory_create_calls / iterations as u64,
            db_opens: warm_operations.db_opens / iterations as u64,
        },
    );
}

fn synthetic(asset_count: usize, iterations: usize) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after Unix epoch")
        .as_nanos();
    let data_dir = std::env::temp_dir().join(format!("sceneworks-assets-list-bench-{nonce}"));
    let store = ProjectStore::new(data_dir.clone(), "assets-list-benchmark");
    let project = store
        .create_project("Synthetic asset-list benchmark")
        .expect("synthetic project creates");
    let project_path = PathBuf::from(&project.path);
    for index in 0..asset_count {
        let asset_id = format!("asset_{index:08}");
        let generation_set_id = format!("set_{index:08}");
        store
            .write_generation_set(
                &project.id,
                "benchmark-job",
                &json!({
                    "id": generation_set_id,
                    "mode": "image_to_video",
                    "model": "benchmark",
                    "prompt": "benchmark",
                    "createdAt": "2026-07-25T00:00:00Z",
                }),
                None,
            )
            .expect("synthetic generation set persists");
        let media_path = format!("assets/videos/{generation_set_id}/{asset_id}.mp4");
        let poster_path = project_path.join(&media_path).with_extension("poster.jpg");
        std::fs::create_dir_all(poster_path.parent().expect("poster parent"))
            .expect("poster directory creates");
        std::fs::write(&poster_path, b"benchmark-poster").expect("poster writes");
        store
            .persist_generated_asset(
                &project.id,
                "benchmark-job",
                &generation_set_id,
                &json!({
                    "assetId": asset_id,
                    "mediaPath": media_path,
                    "mimeType": "video/mp4",
                    "type": "video",
                    "displayName": format!("Asset {index}"),
                    "createdAt": format!("2026-07-25T00:00:00Z-{index:08}"),
                    "mode": "image_to_video",
                    "model": "benchmark",
                    "adapter": "benchmark",
                    "prompt": "benchmark",
                }),
            )
            .expect("synthetic asset persists");
    }
    println!(
        "storage=synthetic-local-distinct-video-sets path={}",
        data_dir.display()
    );
    // Seed through one store, then report the first call through a genuinely
    // fresh registry cache. This still does not evict the operating-system cache.
    run(
        ProjectStore::new(data_dir, "assets-list-benchmark"),
        &project.id,
        iterations,
    );
}

fn existing(data_dir: PathBuf, project_id: &str, iterations: usize) {
    println!(
        "storage=existing path={} project_id={project_id}",
        data_dir.display()
    );
    run(
        ProjectStore::new(data_dir, "assets-list-benchmark"),
        project_id,
        iterations,
    );
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  assets_list_benchmark synthetic [asset_count] [warm_iterations]\n  \
         assets_list_benchmark existing <data_dir> <project_id> [warm_iterations]"
    );
    std::process::exit(2);
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("synthetic") => {
            let asset_count = args
                .get(1)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1_000);
            let iterations = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(10)
                .max(1);
            synthetic(asset_count, iterations);
        }
        Some("existing") if args.len() >= 3 => {
            let iterations = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(10)
                .max(1);
            existing(PathBuf::from(&args[1]), &args[2], iterations);
        }
        _ => usage(),
    }
}
