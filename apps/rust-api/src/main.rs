/// CPU-only proof entry, intentionally before Iris, GPU checks, tracing, worker and API startup.
fn flux3_witness_entry(
    args: &[std::ffi::OsString],
    visibility: Option<&str>,
    owned_root: Option<&std::ffi::OsStr>,
) -> Result<serde_json::Value, String> {
    use sha2::{Digest, Sha256};
    if args.len() != 2 || args[0] != "--flux3-training-config-witness" || visibility != Some("-1") {
        return Err("refused".to_owned());
    }
    let root =
        std::fs::canonicalize(owned_root.ok_or("refused")?).map_err(|_| "refused".to_owned())?;
    let file = std::fs::canonicalize(&args[1]).map_err(|_| "refused".to_owned())?;
    if file == root || !file.starts_with(&root) {
        return Err("refused".to_owned());
    }
    let metadata = std::fs::metadata(&file).map_err(|_| "refused".to_owned())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 2 * 1024 * 1024 {
        return Err("refused".to_owned());
    }
    let bytes = std::fs::read(&file).map_err(|_| "refused".to_owned())?;
    if bytes.len() as u64 != metadata.len() {
        return Err("refused".to_owned());
    }
    let plan: sceneworks_core::training::TrainingPlan =
        serde_json::from_slice(&bytes).map_err(|_| "refused".to_owned())?;
    let mut witness = sceneworks_worker::flux3_training_config_witness(&plan).map_err(|error| {
        if error.starts_with("unsupported:") {
            "unsupported".to_owned()
        } else {
            "refused".to_owned()
        }
    })?;
    witness["inputSha256"] = serde_json::json!(format!("{:x}", Sha256::digest(&bytes)));
    if serde_json::to_vec(&witness).map_err(|_| "refused")?.len() > 1024 * 1024 {
        return Err("refused".to_owned());
    }
    Ok(witness)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args
        .iter()
        .any(|arg| arg == "--flux3-training-config-witness")
    {
        let result = flux3_witness_entry(
            &args,
            std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
            std::env::var_os("SCENEWORKS_FLUX3_WITNESS_ROOT").as_deref(),
        );
        match result {
            Ok(witness) => {
                println!("{}", serde_json::json!({"status":"ok","witness":witness}));
                std::process::exit(0);
            }
            Err(code) => {
                println!(
                    "{}",
                    serde_json::json!({"status":"error","formatVersion":1,"code":code})
                );
                std::process::exit(2);
            }
        }
    }
    // A multi-rank Iris training job (sc-25689) re-enters this same binary once per rank (the MLX
    // worker IS this binary); a rank process trains its rank and exits before anything else — the
    // GPU check, the worker loop or the API — starts. Checked after the model-less witness:
    // rank processes inherit the
    // worker's environment, including SCENEWORKS_WORKER_ONLY.
    if let Some(code) = sceneworks_worker::iris_rank_entry() {
        std::process::exit(code);
    }
    // The desktop app launches this same binary a second time as a standalone
    // GPU worker — the Apple-Silicon MLX worker (sc-3289) — by setting
    // SCENEWORKS_WORKER_ONLY=1. The binary already links the mlx-gen engine, so
    // reusing it avoids bundling a second multi-hundred-MB sidecar.
    // One-shot Metal preflight (sc-8411): the desktop re-launches this binary with
    // SCENEWORKS_GPU_CHECK=1 at startup to verify a usable Metal GPU before spawning
    // the long-running worker. Print the user-facing reason to stdout (the desktop
    // relays it onto the setup screen) and exit non-zero on failure, rather than
    // returning Err (whose Debug print would be noise). Checked before WORKER_ONLY so
    // the probe never starts a worker loop.
    // Exit code carries the SEVERITY, stdout the reason (sc-16247): 0 = usable GPU, 2 = a
    // definite host-side problem the desktop must stop on, 1 = the probe failed for a reason
    // that may not be fatal (a transient CUDA OOM, an unrecognized error), which the desktop
    // logs and steps over rather than locking the user out of the whole app. macOS keeps its
    // sc-8411 contract: every Metal failure is blocking, so it always exits 2 here.
    if std::env::var("SCENEWORKS_GPU_CHECK").is_ok_and(|value| value.trim() == "1") {
        match sceneworks_rust_api::gpu_check() {
            Ok(()) => std::process::exit(0),
            Err(failure) => {
                println!("{}", failure.message);
                std::process::exit(if failure.blocking { 2 } else { 1 });
            }
        }
    }
    if std::env::var("SCENEWORKS_WORKER_ONLY").is_ok_and(|value| value.trim() == "1") {
        return sceneworks_worker::run_on_worker_entry_thread(|| async {
            sceneworks_rust_api::run_worker()
                .await
                .map_err(|error| sceneworks_worker::WorkerError::Engine(error.to_string()))
        })
        .map_err(Into::into);
    }
    sceneworks_rust_api::api_runtime()?.block_on(sceneworks_rust_api::run())
}

#[cfg(test)]
mod flux3_witness_tests {
    use super::flux3_witness_entry;
    use std::ffi::OsString;

    #[test]
    fn witness_refuses_visibility_arguments_and_unowned_or_unbounded_plan() {
        let root = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let owned = root.path().join("plan.json");
        std::fs::write(&owned, br#"{"caption":"private fixture text"}"#).unwrap();
        let args = vec![
            OsString::from("--flux3-training-config-witness"),
            owned.into(),
        ];
        let call = |args: &[OsString], visibility| {
            flux3_witness_entry(args, visibility, Some(root.path().as_os_str()))
        };
        for visibility in [None, Some(""), Some("1")] {
            assert_eq!(call(&args, visibility).unwrap_err(), "refused");
        }
        assert_eq!(call(&args[..1], Some("-1")).unwrap_err(), "refused");
        assert_eq!(call(&args, Some("-1")).unwrap_err(), "refused");
        let foreign = other.path().join("plan.json");
        std::fs::write(&foreign, b"{}").unwrap();
        let foreign_args = vec![args[0].clone(), foreign.into()];
        assert_eq!(call(&foreign_args, Some("-1")).unwrap_err(), "refused");
        let large = root.path().join("large.json");
        std::fs::write(&large, vec![b' '; 2 * 1024 * 1024 + 1]).unwrap();
        let large_args = vec![args[0].clone(), large.into()];
        assert_eq!(call(&large_args, Some("-1")).unwrap_err(), "refused");
        assert_eq!(
            flux3_witness_entry(&args, Some("-1"), None).unwrap_err(),
            "refused"
        );
    }
}
