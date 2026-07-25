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
        "{label}: assets={assets} elapsed_ms={:.3} fs_total={} sidecar_reads={} \
         generation_set_reads={} poster_stats={} directory_create_calls={} db_opens={}",
        elapsed.as_secs_f64() * 1_000.0,
        operations.total(),
        operations.sidecar_reads,
        operations.generation_set_reads,
        operations.poster_stats,
        operations.directory_create_calls,
        operations.db_opens,
    );
}

fn run(store: ProjectStore, project_id: &str, iterations: usize) {
    let (assets, cold_elapsed, cold_operations) = list_once(&store, project_id);
    report("cold", assets, cold_elapsed, cold_operations);

    let mut warm_elapsed = Duration::ZERO;
    let mut warm_operations = AssetListFilesystemOperations::default();
    for _ in 0..iterations {
        let (warm_assets, elapsed, operations) = list_once(&store, project_id);
        assert_eq!(warm_assets, assets, "asset count changed during benchmark");
        warm_elapsed += elapsed;
        warm_operations.sidecar_reads += operations.sidecar_reads;
        warm_operations.generation_set_reads += operations.generation_set_reads;
        warm_operations.poster_stats += operations.poster_stats;
        warm_operations.directory_create_calls += operations.directory_create_calls;
        warm_operations.db_opens += operations.db_opens;
    }
    let divisor = iterations as u32;
    report(
        &format!("warm-average({iterations})"),
        assets,
        warm_elapsed / divisor,
        AssetListFilesystemOperations {
            sidecar_reads: warm_operations.sidecar_reads / iterations as u64,
            generation_set_reads: warm_operations.generation_set_reads / iterations as u64,
            poster_stats: warm_operations.poster_stats / iterations as u64,
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
    for index in 0..asset_count {
        let asset_id = format!("asset_{index:08}");
        store
            .persist_generated_asset(
                &project.id,
                "benchmark-job",
                "shared-set",
                &json!({
                    "assetId": asset_id,
                    "mediaPath": format!("assets/images/shared-set/{asset_id}.png"),
                    "mimeType": "image/png",
                    "displayName": format!("Asset {index}"),
                    "createdAt": format!("2026-07-25T00:00:00Z-{index:08}"),
                    "mode": "text_to_image",
                    "model": "benchmark",
                    "adapter": "benchmark",
                    "prompt": "benchmark",
                }),
            )
            .expect("synthetic asset persists");
    }
    println!("storage=synthetic-local path={}", data_dir.display());
    run(store, &project.id, iterations);
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
