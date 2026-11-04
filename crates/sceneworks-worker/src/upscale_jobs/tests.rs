//! Pure-math unit tests for the Real-ESRGAN tiling port (sc-3489). No onnx weights
//! needed — these lock the tiling/crop/manifest logic against `upscalers.py`. They run on
//! macOS and on the off-Mac candle lane (sc-5499), since the tiling path is now shared.

use super::*;

#[test]
fn ort_upscaler_cache_key_includes_factor_and_resolved_path() {
    assert_eq!(
        upscaler_cache_key(2, Path::new("/weights/a.onnx")),
        upscaler_cache_key(2, Path::new("/weights/a.onnx"))
    );
    assert_ne!(
        upscaler_cache_key(2, Path::new("/weights/a.onnx")),
        upscaler_cache_key(2, Path::new("/weights/b.onnx"))
    );
    assert_ne!(
        upscaler_cache_key(2, Path::new("/weights/a.onnx")),
        upscaler_cache_key(4, Path::new("/weights/a.onnx"))
    );
}
// `Rgb`/`RgbImage` back the Real-ESRGAN tiling/crop tests + the off-Mac ort smoke (the `ort` path,
// macOS + the candle lane) AND the SeedVR2 real-weight smoke (Mac MLX + the candle lane).
#[cfg(native_lane)]
use image::{Rgb, RgbImage};
use serde_json::json;

#[test]
fn onnx_filename_per_factor() {
    assert_eq!(onnx_file(2), "real_esrgan_x2.onnx");
    assert_eq!(onnx_file(4), "real_esrgan_x4.onnx");
}

/// F-118 / sc-9794 follow-up: image upscale accepts only 2x and 4x. Any other factor is rejected
/// with a clear `InvalidPayload` error rather than silently coerced to 2x (which produced a
/// quietly-different output), mirroring the video path's `resolve_video_upscale_factor`
/// (F-118 / sc-8920). Both `run_image_upscale_job` and `run_dataset_upscale_job` route through
/// this guard.
#[test]
fn image_upscale_factor_accepts_2_and_4_rejects_others() {
    assert_eq!(resolve_image_upscale_factor(2).expect("2x"), 2);
    assert_eq!(resolve_image_upscale_factor(4).expect("4x"), 4);
    for bad in [0u64, 1, 3, 5, 8, 16] {
        let err = resolve_image_upscale_factor(bad)
            .expect_err("unsupported factor must be rejected, not coerced");
        assert!(
            matches!(err, WorkerError::InvalidPayload(ref m) if m.contains("factor 2 or 4")),
            "factor {bad} should yield a clear InvalidPayload error, got {err:?}"
        );
    }
}

#[cfg(native_lane)]
#[test]
fn manifest_onnx_resource_extracts_repo_file() {
    let entry = json!({
        "resources": {
            "imageUpscalers": {
                "real-esrgan": {
                    "x4": { "onnx": { "repo": "acme/esrgan-onnx", "file": "x4.onnx" } }
                }
            }
        }
    });
    assert_eq!(
        manifest_onnx_resource(&entry, 4),
        Some(("acme/esrgan-onnx".to_owned(), "x4.onnx".to_owned()))
    );
    // missing factor → None (falls back to default repo)
    assert_eq!(manifest_onnx_resource(&entry, 2), None);
    // file defaults to the conventional name when absent
    let no_file = json!({
        "resources": { "imageUpscalers": { "real-esrgan": { "x2": { "onnx": { "repo": "acme/e" } } } } }
    });
    assert_eq!(
        manifest_onnx_resource(&no_file, 2),
        Some(("acme/e".to_owned(), "real_esrgan_x2.onnx".to_owned()))
    );
    assert_eq!(manifest_onnx_resource(&Value::Null, 4), None);
}

// ---------------------------------------------------------------------------
// SeedVR2 (sc-4815): pure helpers + a gated real-weight integration smoke
// ---------------------------------------------------------------------------

#[test]
fn round_to_16_rounds_up_floored_at_16() {
    assert_eq!(round_to_16(96), 96); // already a multiple
    assert_eq!(round_to_16(64), 64);
    assert_eq!(round_to_16(100), 112); // rounds up to the next multiple of 16
    assert_eq!(round_to_16(17), 32);
    assert_eq!(round_to_16(1), 16); // floored at 16
    assert_eq!(round_to_16(0), 16);
}

#[test]
fn upscale_target_dimensions_are_bounded_before_allocation() {
    validate_upscale_target_dimensions(8192, 8192).expect("8k square accepted");
    assert!(matches!(
        validate_upscale_target_dimensions(8193, 1024),
        Err(WorkerError::InvalidPayload(_))
    ));
    assert!(matches!(
        validate_upscale_target_dimensions(8192, 8193),
        Err(WorkerError::InvalidPayload(_))
    ));
}

/// Resolve the locally-cached `numz/SeedVR2_comfyUI` checkpoint dir (env override or the HF cache),
/// so the smoke below can run on real weights without a download. `None` ⇒ skip.
#[cfg(native_lane)]
fn cached_seedvr2_checkpoint() -> Option<std::path::PathBuf> {
    if let Ok(pinned) = std::env::var("SCENEWORKS_SEEDVR2_CHECKPOINT") {
        let dir = std::path::PathBuf::from(pinned);
        if dir.join(SEEDVR2_DIT_FILE).exists() && dir.join(SEEDVR2_VAE_FILE).exists() {
            return Some(dir);
        }
    }
    let base = std::path::Path::new(&std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    (snap.join(SEEDVR2_DIT_FILE).exists() && snap.join(SEEDVR2_VAE_FILE).exists()).then_some(snap)
}

/// Real-weight smoke for the SceneWorks SeedVR2 integration (sc-4815 Mac / sc-5928 candle): drives the
/// exact worker dispatch path — `with_cached_generator("seedvr2", …)` → registry → `generate` — on the
/// cached 3B checkpoint, asserting (a) the factor→`round_to_16` target dims and (b) that the `softness`
/// request field actually reaches the engine (a softened run differs from a faithful one). On Mac this
/// resolves to the MLX provider; on the Windows/CUDA candle build it resolves to `candle-gen-seedvr2`
/// (the sc-5928 worker-path validation that the bundle catalog + routing reach the candle engine end-to-
/// end). Gated on the checkpoint being present (skips in CI, which has no weights), mirroring the
/// family worker E2E smokes. Set `SCENEWORKS_SEEDVR2_CHECKPOINT` to the ckpt dir and run with
/// `cargo test -p sceneworks-worker --features backend-candle -- --ignored seedvr2_upscale_real_weight_smoke`.
#[cfg(native_lane)]
#[tokio::test]
#[ignore = "real-weight: needs the cached numz/SeedVR2_comfyUI checkpoint (~7 GB) + the seedvr2 backend (MLX on Mac / candle on Windows)"]
async fn seedvr2_upscale_real_weight_smoke() {
    let Some(dir) = cached_seedvr2_checkpoint() else {
        eprintln!("SKIP: SeedVR2 checkpoint not cached (numz/SeedVR2_comfyUI)");
        return;
    };
    // 48x32 deterministic gradient → factor 2 → 96x64 (both already multiples of 16).
    let mut img = RgbImage::new(48, 32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([(x * 5) as u8, (y * 7) as u8, ((x + y) * 3 % 256) as u8]);
    }
    let faithful = run_seedvr2_upscale(dir.clone(), img.clone(), 2, 0.0, 7, CancelFlag::new())
        .await
        .expect("seedvr2 upscale (softness 0)");
    assert_eq!((faithful.width(), faithful.height()), (96, 64));

    // The softness request field must reach the engine: a heavily-softened run changes the result.
    let softened = run_seedvr2_upscale(dir, img, 2, 0.8, 7, CancelFlag::new())
        .await
        .expect("seedvr2 upscale (softness 0.8)");
    assert_eq!((softened.width(), softened.height()), (96, 64));
    assert_ne!(
        faithful.as_raw(),
        softened.as_raw(),
        "softness must change the output (the request field is wired to the engine)"
    );
}

// ---------------------------------------------------------------------------
// Real-ESRGAN off-Mac (sc-5499): a gated real-weight smoke on the candle lane's `ort`/CUDA path
// ---------------------------------------------------------------------------

/// Real-weight smoke for the off-Mac Real-ESRGAN `ort` path (sc-5499): drives the exact worker seam
/// `upscale_blocking` → `Upscaler::load` (CUDA EP, CPU fallback) → tiled `Upscaler::upscale` on a real
/// exported `real_esrgan_x{2,4}.onnx`, asserting the output is exactly `factor×` the input and is a
/// real super-resolution (not an all-zero / identity buffer). Gated on the ONNX being present (skips in
/// CI, which has no weights), like the SeedVR2 smoke. Point `SCENEWORKS_REALESRGAN_X{factor}_ONNX` at a
/// local export (optionally `SCENEWORKS_TEST_UPSCALE_IMAGE` at a source image; else a synthetic gradient
/// is used) and run with `cargo test -p sceneworks-worker --features backend-candle --lib -- --ignored
/// real_esrgan_candle_real_weights_upscales`. `ORT_DYLIB_PATH` must point at an onnxruntime build.
#[test]
#[ignore = "real-weight: needs an exported real_esrgan ONNX (SCENEWORKS_REALESRGAN_X{factor}_ONNX) + an onnxruntime (ORT_DYLIB_PATH)"]
fn real_esrgan_candle_real_weights_upscales() {
    let factor: u8 = match std::env::var("SCENEWORKS_TEST_UPSCALE_FACTOR")
        .ok()
        .as_deref()
    {
        Some("4") => 4,
        _ => 2,
    };
    let Some(onnx) = ["SCENEWORKS_REALESRGAN_X".to_owned() + &factor.to_string() + "_ONNX"]
        .into_iter()
        .chain(["SCENEWORKS_REALESRGAN_ONNX".to_owned()])
        .find_map(|key| std::env::var(&key).ok().map(std::path::PathBuf::from))
        .filter(|p| p.exists())
    else {
        panic!("required Real-ESRGAN x{factor} ONNX proof fixture is missing");
    };

    // A real photo if provided, else a deterministic gradient (Real-ESRGAN is a deterministic conv,
    // so a synthetic input still exercises the full graph + tiling end-to-end).
    let src = match std::env::var("SCENEWORKS_TEST_UPSCALE_IMAGE").ok() {
        Some(path) => image::open(&path)
            .unwrap_or_else(|e| panic!("load SCENEWORKS_TEST_UPSCALE_IMAGE {path}: {e}"))
            .to_rgb8(),
        None => {
            let mut img = RgbImage::new(513, 33);
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                *pixel = Rgb([(x * 3) as u8, (y * 5) as u8, ((x + y) * 2 % 256) as u8]);
            }
            img
        }
    };
    let (sw, sh) = (src.width(), src.height());

    // `upscale_blocking` now returns the image plus the `ort` session's execution device
    // (coreml/cuda/cpu, sc-8923); the smoke only checks the image, so bind the device for the trace.
    let (out, device) = upscale_blocking(onnx.clone(), factor, src, CancelFlag::new())
        .expect("Real-ESRGAN upscale");
    assert_eq!(
        (out.width(), out.height()),
        (sw * u32::from(factor), sh * u32::from(factor)),
        "output must be exactly factor× the source"
    );
    // not an all-zero buffer (the graph actually ran + wrote pixels).
    assert!(
        out.as_raw().iter().any(|&b| b != 0),
        "upscaled image must not be all-black"
    );
    eprintln!(
        "Real-ESRGAN x{factor} ({device}): {sw}x{sh} -> {}x{}",
        out.width(),
        out.height()
    );
    if let Ok(directory) = std::env::var("CV_PROOF_OUTPUT") {
        out.save(Path::new(&directory).join("upscaled.png"))
            .expect("save CV upscale evidence");
    }
    if std::env::var_os("CV_PROOF_OUTPUT").is_some() {
        let profiles = UPSCALERS
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .get_mut(&upscaler_cache_key(factor, &onnx))
            .unwrap()
            .finish_profiling()
            .unwrap();
        crate::cv::proof_ort_complete(
            "real_esrgan_candle_real_weights_upscales",
            device,
            json!({"factor": factor, "width": out.width(), "height": out.height(),
                "sourceWidth": sw, "sourceHeight": sh, "tileSize": 512, "tilePad": 16}),
            profiles,
        );
    }
}

// --- Dataset Doctor one-tap upscale plumbing (sc-6539), pure + weight-free ---

#[cfg(test)]
fn upscale_test_settings(data_dir: &std::path::Path) -> crate::Settings {
    crate::Settings {
        api_url: "http://127.0.0.1".to_owned(),
        config_dir: data_dir.join("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        huggingface_base_url: crate::DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
        max_lora_url_bytes: crate::DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: crate::DEFAULT_MAX_MODEL_URL_BYTES,
        backend_mlx_enabled: true,
        ..crate::Settings::for_test(data_dir.to_path_buf())
    }
}

#[test]
fn parse_dataset_upscale_items_reads_valid_entries_and_skips_malformed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings = upscale_test_settings(dir.path());
    let dataset_root = dir.path().join("datasets").join("ds-1");
    let a = dataset_root.join("a.png");
    let d = dataset_root.join("d.jpg");
    let payload = json!({
        "datasetRoot": dataset_root.display().to_string(),
        "items": [
            { "itemId": "a", "imagePath": a.display().to_string(), "assetId": "asset_a" },
            { "itemId": "", "imagePath": dataset_root.join("blank.png").display().to_string() }, // empty id → skipped
            { "itemId": "c" },                                                                    // no imagePath → skipped
            { "itemId": "d", "imagePath": d.display().to_string() },
        ],
    });
    let items =
        parse_dataset_upscale_items(&settings, payload.as_object().unwrap()).expect("items parse");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].item_id, "a");
    // sc-9812: path confinement canonicalizes the deepest existing ancestor before
    // re-appending the (not-yet-created) tail, so the resolved image path is expressed
    // via the canonical tempdir root (on macOS `/var` -> `/private/var`).
    let expected_a = dir
        .path()
        .canonicalize()
        .expect("tempdir canonicalizes")
        .join("datasets")
        .join("ds-1")
        .join("a.png");
    assert_eq!(items[0].image_path, expected_a);
    assert_eq!(items[1].item_id, "d");
}

#[test]
fn parse_dataset_upscale_items_is_empty_without_an_items_array() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings = upscale_test_settings(dir.path());
    let dataset_root = dir.path().join("datasets").join("ds-1");
    let payload = json!({ "datasetRoot": dataset_root.display().to_string() });
    assert!(
        parse_dataset_upscale_items(&settings, payload.as_object().unwrap())
            .expect("empty parse")
            .is_empty()
    );
}

// sc-8842 / F-040: a client-supplied `imagePath` escaping the app-managed dataset root is an
// arbitrary-file-read/exfil primitive; `resolve_dataset_item_path` must reject it so the job fails
// rather than reading (and later re-pointing at) an out-of-tree file.
#[test]
fn parse_dataset_upscale_items_reject_image_path_outside_dataset_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings = upscale_test_settings(dir.path());
    let dataset_root = dir.path().join("datasets").join("ds-1");
    let payload = json!({
        "datasetRoot": dataset_root.display().to_string(),
        "items": [
            {
                "itemId": "item_1",
                "imagePath": dataset_root.join("../../etc/secret.png").display().to_string(),
            },
        ],
    });
    let error = parse_dataset_upscale_items(&settings, payload.as_object().unwrap())
        .expect_err("traversal image path rejected");
    assert!(
        error
            .to_string()
            .contains("Dataset upscale item item_1 imagePath"),
        "{error}"
    );
}

#[test]
fn parse_dataset_upscale_items_reject_missing_dataset_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings = upscale_test_settings(dir.path());
    let payload = json!({ "items": [{ "itemId": "a", "imagePath": "a.png" }] });
    let error = parse_dataset_upscale_items(&settings, payload.as_object().unwrap())
        .expect_err("missing datasetRoot rejected");
    assert!(error.to_string().contains("datasetRoot"), "{error}");
}

// sc-8842 / F-040: the output write path interpolates client-supplied `datasetId`/`itemId`; a
// traversal component must be rejected by `safe_project_path` before `save_with_format` runs so a
// crafted id cannot overwrite an arbitrary out-of-tree file with PNG bytes.
#[test]
fn dataset_upscale_output_path_rejects_traversal_ids() {
    let dir = tempfile::tempdir().expect("tempdir");
    let project_path = dir.path().join("project");
    std::fs::create_dir(&project_path).expect("project root exists before dataset output");
    // itemId traversal
    let rel = format!("training/datasets/ds-1/upscaled/{}.png", "../../escaped");
    let error =
        crate::safe_project_path(&project_path, &rel).expect_err("traversal itemId rejected");
    assert!(
        error.to_string().contains("Unsafe project-relative path"),
        "{error}"
    );
    // datasetId traversal
    let rel = format!("training/datasets/{}/upscaled/a.png", "../../../etc");
    let error =
        crate::safe_project_path(&project_path, &rel).expect_err("traversal datasetId rejected");
    assert!(
        error.to_string().contains("Unsafe project-relative path"),
        "{error}"
    );
    // valid ids resolve inside the project tree
    let rel = "training/datasets/ds-1/upscaled/item_1.png";
    let abs = crate::safe_project_path(&project_path, rel).expect("valid path resolves");
    let canonical_project = project_path.canonicalize().expect("project root resolves");
    assert_eq!(abs, canonical_project.join(rel));
    assert!(
        abs.ends_with("training/datasets/ds-1/upscaled/item_1.png"),
        "{abs:?}"
    );
}

/// sc-8911: a set-but-missing Real-ESRGAN ONNX env pin must error, not silently fall
/// through to the cache/HF download. Unset → `None`; set + existing → `Some(path)`.
#[test]
fn resolve_env_file_pin_errors_on_missing_path() {
    use std::ffi::OsString;

    assert_eq!(
        resolve_env_file_pin("SCENEWORKS_REALESRGAN_ONNX", None, "the export").expect("unset ok"),
        None,
        "an unset pin must fall through"
    );

    let missing = resolve_env_file_pin(
        "SCENEWORKS_REALESRGAN_X4_ONNX",
        Some(OsString::from("/nonexistent/realesrgan_x4.onnx")),
        "the local Real-ESRGAN x4 ONNX export",
    );
    assert!(
        matches!(missing, Err(WorkerError::InvalidPayload(ref m)) if m.contains("SCENEWORKS_REALESRGAN_X4_ONNX") && m.contains("does not exist")),
        "a set-but-missing pin must error, got {missing:?}"
    );

    // The pin names a FILE, so the guard is the directory around it (sc-17707).
    let pin_guard = tempfile::Builder::new()
        .prefix("sw-realesrgan-pin-test-")
        .tempdir()
        .expect("temp dir");
    let existing = pin_guard.path().join("realesrgan_x4.onnx");
    std::fs::write(&existing, b"onnx").expect("write temp onnx");
    let resolved = resolve_env_file_pin(
        "SCENEWORKS_REALESRGAN_ONNX",
        Some(existing.as_os_str().to_owned()),
        "the export",
    )
    .expect("existing pin ok");
    assert_eq!(resolved.as_deref(), Some(existing.as_path()));
}

/// Unset falls through and a complete dir resolves — for BOTH keys, whatever their strictness.
/// The incomplete case is where they diverge, and it is covered by
/// `incomplete_pins_split_on_which_key_carried_them` below.
#[test]
fn resolve_seedvr2_dir_pin_accepts_a_complete_dir_and_ignores_an_unset_one() {
    use std::ffi::OsString;

    assert_eq!(
        SEEDVR2_DIR_PINS,
        [
            ("SCENEWORKS_SEEDVR2_CHECKPOINT", SeedVr2Pin::NamedCheckpoint),
            ("SCENEWORKS_SEEDVR2_DIR", SeedVr2Pin::StagingDir),
        ],
        "both lanes' historical pins must be honored, each with its own strictness"
    );
    for (key, kind) in SEEDVR2_DIR_PINS {
        assert_eq!(
            resolve_seedvr2_dir_pin(key, *kind, None).expect("unset ok"),
            None,
            "an unset {key} must fall through"
        );

        let guard = tempfile::Builder::new()
            .prefix("sw-seedvr2-pin-test-")
            .tempdir()
            .expect("temp dir");
        let dir = guard.path();
        std::fs::write(dir.join(SEEDVR2_DIT_FILE), b"dit").expect("write dit");
        std::fs::write(dir.join(SEEDVR2_VAE_FILE), b"vae").expect("write vae");
        let resolved = resolve_seedvr2_dir_pin(key, *kind, Some(OsString::from(dir.as_os_str())))
            .expect("complete dir ok");
        assert_eq!(resolved.as_deref(), Some(dir));
    }
}

/// sc-17632 review — a set-but-INCOMPLETE pin means different things per key, because the two keys
/// meant different things before this story collapsed the lanes onto one resolver.
///
/// `SCENEWORKS_SEEDVR2_CHECKPOINT` NAMES a checkpoint, so incomplete is an operator error and stays
/// sc-8911's loud failure — naming the key AND the files actually missing, not "X and/or Y".
/// `SCENEWORKS_SEEDVR2_DIR` was a download DESTINATION the worker created and staged into, so an
/// empty one is the state it was in before first use; hard-failing on it would be a rule this story
/// invented, and would let a stale export make a correctly installed model unreachable.
///
/// Asserted per key by NAME, not by iterating the table: a table that lost the `NamedCheckpoint`
/// row entirely would still satisfy a loop.
#[test]
fn incomplete_pins_split_on_which_key_carried_them() {
    use std::ffi::OsString;

    let guard = tempfile::Builder::new()
        .prefix("sw-seedvr2-partial-")
        .tempdir()
        .expect("temp dir");
    let dir = guard.path();
    // Half-populated: the DiT is there, the VAE is not. The error must say so precisely.
    std::fs::write(dir.join(SEEDVR2_DIT_FILE), b"dit").expect("write dit");
    let raw = || Some(OsString::from(dir.as_os_str()));

    let strict = resolve_seedvr2_dir_pin(
        "SCENEWORKS_SEEDVR2_CHECKPOINT",
        SeedVr2Pin::NamedCheckpoint,
        raw(),
    );
    let message = match &strict {
        Err(WorkerError::InvalidPayload(message)) => message.clone(),
        other => panic!("an incomplete named-checkpoint pin must error, got {other:?}"),
    };
    assert!(
        message.contains("SCENEWORKS_SEEDVR2_CHECKPOINT"),
        "the error must name the key the operator set, got {message}"
    );
    assert!(
        message.contains(SEEDVR2_VAE_FILE),
        "the error must name the file that is actually missing, got {message}"
    );
    assert!(
        !message.contains(SEEDVR2_DIT_FILE),
        "the error must NOT name a file that is present — that sends the operator looking in the \
         wrong place, got {message}"
    );
    assert!(
        message.contains("unset it"),
        "the error must say how to recover, got {message}"
    );

    assert_eq!(
        resolve_seedvr2_dir_pin("SCENEWORKS_SEEDVR2_DIR", SeedVr2Pin::StagingDir, raw())
            .expect("an incomplete staging dir must not be an error"),
        None,
        "an incomplete SCENEWORKS_SEEDVR2_DIR must fall through to the next candidate"
    );
}

/// sc-8879: the SeedVR2 mirror is read at a pinned commit, never the mutable `main` branch, so an
/// upstream re-push can't silently swap the weights we load. Lock the constant to a real 40-hex
/// commit id. Since sc-17632 this const is the ONLY one — the video lane's verbatim duplicate is
/// gone — so this single check now covers both lanes.
#[test]
fn seedvr2_revision_is_pinned_commit_not_main() {
    assert_ne!(
        SEEDVR2_REVISION, "main",
        "SeedVR2 must pin a fixed revision"
    );
    assert_eq!(
        SEEDVR2_REVISION.len(),
        40,
        "a pinned HF revision is a 40-char commit sha"
    );
    assert!(
        SEEDVR2_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the pinned revision must be lowercase hex"
    );
}

/// sc-9682 (F-077 follow-up): the first-party Real-ESRGAN ONNX repo is fetched at a
/// pinned commit, never the mutable `main` branch, so a re-push (or a compromised token)
/// can't silently swap the ONNX graph we load. Lock the constant to a real 40-hex commit id.
#[test]
fn onnx_revision_is_pinned_commit_not_main() {
    assert_ne!(
        ONNX_REVISION, "main",
        "Real-ESRGAN ONNX must pin a fixed revision"
    );
    assert_eq!(
        ONNX_REVISION.len(),
        40,
        "a pinned HF revision is a 40-char commit sha"
    );
    assert!(
        ONNX_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the pinned revision must be lowercase hex"
    );
}

#[test]
fn dataset_repoint_body_maps_records_with_a_null_asset_id() {
    let body = dataset_repoint_body(&[
        (
            "a".to_owned(),
            "training/datasets/ds/upscaled/a.png".to_owned(),
        ),
        (
            "b".to_owned(),
            "training/datasets/ds/upscaled/b.png".to_owned(),
        ),
    ]);
    let items = body
        .get("items")
        .and_then(serde_json::Value::as_array)
        .expect("items array");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["itemId"], "a");
    assert_eq!(
        items[0]["sourcePath"],
        "training/datasets/ds/upscaled/a.png"
    );
    assert!(
        items[0]["assetId"].is_null(),
        "dataset fix re-points to bytes, not a minted child asset"
    );
}

/// sc-17633 (epic 17625) — the Real-ESRGAN ONNX resolves from the HF cache and can no longer
/// download. Before this, `ensure_onnx` fetched ~67 MB per factor mid-upscale into
/// `<data_dir>/cache/upscale/`, a destination the Models screen can neither size nor delete.
#[cfg(native_lane)]
mod resolve_onnx_tests {
    use super::*;

    fn isolate() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("SCENEWORKS_REALESRGAN_X2_ONNX", ""),
            ("SCENEWORKS_REALESRGAN_X4_ONNX", ""),
            ("SCENEWORKS_REALESRGAN_ONNX", ""),
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    fn settings_at(data_dir: PathBuf) -> crate::Settings {
        let mut settings = crate::test_env::offline_settings();
        settings.data_dir = data_dir;
        settings
    }

    fn stage_install(data_dir: &Path, factor: u8) -> PathBuf {
        let snapshot = crate::huggingface_repo_cache_path(data_dir, ONNX_REPO)
            .expect("repo cache path")
            .join("snapshots")
            .join(ONNX_REVISION);
        std::fs::create_dir_all(&snapshot).expect("mk snapshot");
        let path = snapshot.join(onnx_file(factor));
        std::fs::write(&path, b"onnx").expect("write");
        path
    }

    fn stage_legacy(data_dir: &Path, factor: u8) -> PathBuf {
        let dir = data_dir.join("cache").join("upscale");
        std::fs::create_dir_all(&dir).expect("mk legacy");
        let path = dir.join(onnx_file(factor));
        std::fs::write(&path, b"legacy onnx").expect("write");
        path
    }

    /// The Model Manager install (declared at the SAME pin the loader reads) satisfies the loader.
    #[test]
    fn resolves_the_installed_snapshot_from_the_hf_cache() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let staged = stage_install(dir.path(), 4);
        assert_eq!(
            resolve_onnx(&settings_at(dir.path().to_path_buf()), 4, &Value::Null)
                .expect("resolves"),
            staged
        );
    }

    /// AC10: an existing install keeps working from `<data_dir>/cache/upscale/` and re-downloads
    /// nothing — the bytes the old job-time fetch left behind.
    #[test]
    fn falls_back_to_the_legacy_upscale_cache() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let legacy = stage_legacy(dir.path(), 2);
        assert_eq!(
            resolve_onnx(&settings_at(dir.path().to_path_buf()), 2, &Value::Null)
                .expect("resolves"),
            legacy
        );
    }

    /// The legacy copy drains: once installed properly the HF cache wins.
    #[test]
    fn prefers_the_hf_cache_over_the_legacy_copy() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let staged = stage_install(dir.path(), 4);
        stage_legacy(dir.path(), 4);
        assert_eq!(
            resolve_onnx(&settings_at(dir.path().to_path_buf()), 4, &Value::Null)
                .expect("resolves"),
            staged
        );
    }

    /// A manifest `onnx` resource override is honored — and, being a NON-default repo, is read at
    /// `refs/main` rather than the first-party pin. Staging it under the pin proves the override
    /// path is taken (a resolver ignoring the override would find the default repo's absence).
    #[test]
    fn honors_a_manifest_override_repo() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let entry = serde_json::json!({
            "resources": { "imageUpscalers": { "real-esrgan": {
                "x4": { "onnx": { "repo": "acme/esrgan-onnx", "file": "custom_x4.onnx" } }
            } } }
        });
        let repo_dir = crate::huggingface_repo_cache_path(dir.path(), "acme/esrgan-onnx")
            .expect("repo cache path");
        let revision = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
        let snapshot = repo_dir.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("mk snapshot");
        let staged = snapshot.join("custom_x4.onnx");
        std::fs::write(&staged, b"onnx").expect("write");
        std::fs::create_dir_all(repo_dir.join("refs")).expect("mk refs");
        std::fs::write(repo_dir.join("refs").join("main"), revision).expect("write refs/main");

        assert_eq!(
            resolve_onnx(&settings_at(dir.path().to_path_buf()), 4, &entry).expect("resolves"),
            staged
        );
    }

    /// Nothing installed anywhere → an actionable install error, NOT a download (AC7).
    #[test]
    fn errors_actionably_when_nothing_is_installed() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let error = resolve_onnx(&settings_at(dir.path().to_path_buf()), 4, &Value::Null)
            .expect_err("nothing is installed");
        let message = format!("{error:?}");
        assert!(
            message.contains("Model Manager"),
            "the error must tell the user how to fix it, got {message}"
        );
        assert!(matches!(error, WorkerError::InvalidPayload(_)), "{error:?}");
    }
}

/// sc-17632 (epic 17625) — the SeedVR2 checkpoint resolves from the HF cache and can no longer
/// download, on EITHER lane. Before this, `ensure_seedvr2_checkpoint` fetched ~7.3 GB mid-upscale
/// into `<data_dir>/cache/upscale/seedvr2/`, and `video_jobs::seedvr2` fetched the SAME repo again
/// into `<data_dir>/cache/seedvr2-mlx/` — two copies of one checkpoint, in a tree the Models screen
/// can neither size nor delete.
///
/// Neither legacy destination exists on the machine this was written on, so the AC10 fallbacks are
/// exercised against STAGED fixtures (`stage_legacy`) rather than real bytes; what they pin is the
/// resolution ORDER and the whole-pair-per-root rule, which is what an upgrade depends on.
#[cfg(native_lane)]
mod resolve_seedvr2_tests {
    use super::*;

    /// The HF-cache env vars are neutralized alongside the two dir pins: the resolver now consults
    /// the HF cache, so on any machine that has `numz/SeedVR2_comfyUI` cached — a developer box
    /// with the 7.3 GB checkpoint installed — the "nothing is installed" cases would resolve
    /// `Some(..)` and fail for reasons unrelated to what they test (`test_env.rs` documents the
    /// trap).
    fn isolate() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("SCENEWORKS_SEEDVR2_CHECKPOINT", ""),
            ("SCENEWORKS_SEEDVR2_DIR", ""),
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    fn settings_at(data_dir: PathBuf) -> crate::Settings {
        let mut settings = crate::test_env::offline_settings();
        settings.data_dir = data_dir;
        settings
    }

    /// Stage the pinned snapshot the way a `seedvr2_upscaler` Model Manager install leaves it.
    fn stage_install(data_dir: &Path, files: &[&str]) -> PathBuf {
        let snapshot = crate::huggingface_repo_cache_path(data_dir, SEEDVR2_REPO)
            .expect("repo cache path")
            .join("snapshots")
            .join(SEEDVR2_REVISION);
        std::fs::create_dir_all(&snapshot).expect("mk snapshot");
        for file in files {
            std::fs::write(snapshot.join(file), b"hf weights").expect("write");
        }
        snapshot
    }

    fn stage_legacy(data_dir: &Path, sub: &str, files: &[&str]) -> PathBuf {
        let dir = data_dir.join(sub);
        std::fs::create_dir_all(&dir).expect("mk legacy");
        for file in files {
            std::fs::write(dir.join(file), b"legacy weights").expect("write");
        }
        dir
    }

    /// The Model Manager install (declared at the SAME pin the loader reads) satisfies the loader,
    /// and what comes back is the snapshot DIR — what `Seedvr2Pipeline::load` is handed.
    #[test]
    fn resolves_the_installed_snapshot_dir_from_the_hf_cache() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = stage_install(dir.path(), &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]);
        assert_eq!(
            resolve_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()))
                .expect("resolves"),
            Some(snapshot)
        );
    }

    /// AC10, BOTH pre-migration roots: an existing install of EITHER lane keeps working and
    /// re-downloads nothing. The image lane wrote `cache/upscale/seedvr2/`, the video lane
    /// `cache/seedvr2-mlx/`; collapsing onto one resolver means either copy now serves both jobs.
    ///
    /// The population is asserted LITERALLY before the loop, not just iterated: emptying
    /// `SEEDVR2_LEGACY_DIRS` — which is exactly the regression that would silently re-download 7.3
    /// GB on every existing install — makes a bare `for` loop pass vacuously. Verified by mutation.
    #[test]
    fn falls_back_to_either_legacy_root() {
        assert_eq!(
            SEEDVR2_LEGACY_DIRS,
            ["cache/upscale/seedvr2", "cache/seedvr2-mlx"],
            "both lanes' pre-migration destinations must stay readable (AC10)"
        );
        for sub in SEEDVR2_LEGACY_DIRS {
            let _env = isolate();
            let dir = tempfile::tempdir().expect("tempdir");
            let legacy = stage_legacy(dir.path(), sub, &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]);
            assert_eq!(
                resolve_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()))
                    .expect("resolves"),
                Some(legacy),
                "the legacy root {sub} must still satisfy the loader"
            );
        }
    }

    /// The legacy copies drain: once installed properly the HF cache wins even with both old trees
    /// still on disk. Swapping the arms passes the previous test and fails this one.
    #[test]
    fn prefers_the_hf_cache_over_the_legacy_roots() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = stage_install(dir.path(), &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]);
        for sub in SEEDVR2_LEGACY_DIRS {
            stage_legacy(dir.path(), sub, &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]);
        }
        assert_eq!(
            resolve_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()))
                .expect("resolves"),
            Some(snapshot)
        );
    }

    /// A TORN install is not a resolution. The engine is handed a DIRECTORY, so a resolver that
    /// checked the two files independently rather than per root could hand back a snapshot dir
    /// holding only the DiT while the VAE sat in a legacy tree — and the failure would surface deep
    /// inside `Seedvr2Pipeline::load`, not here.
    #[test]
    fn refuses_a_torn_pair_across_roots() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        stage_install(dir.path(), &[SEEDVR2_DIT_FILE]);
        stage_legacy(dir.path(), SEEDVR2_LEGACY_DIRS[0], &[SEEDVR2_VAE_FILE]);
        assert_eq!(
            resolve_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()))
                .expect("resolves"),
            None,
            "a DiT in the HF cache and a VAE in a legacy root is not a usable checkpoint"
        );
    }

    /// Both historical dir pins win over everything installed, and both work on both lanes now
    /// (`SCENEWORKS_SEEDVR2_DIR` used to be inert on the image lane and vice versa). The population
    /// is asserted literally for the same reason as `falls_back_to_either_legacy_root`: dropping a
    /// key would silently make the loop cover less.
    #[test]
    fn either_dir_pin_wins_over_the_hf_cache() {
        assert_eq!(
            SEEDVR2_DIR_PINS
                .iter()
                .map(|(key, _)| *key)
                .collect::<Vec<_>>(),
            ["SCENEWORKS_SEEDVR2_CHECKPOINT", "SCENEWORKS_SEEDVR2_DIR"],
            "both lanes' advertised checkpoint-dir knobs must keep working"
        );
        for (key, _) in SEEDVR2_DIR_PINS {
            let _env = isolate();
            let dir = tempfile::tempdir().expect("tempdir");
            stage_install(dir.path(), &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]);
            let pinned = stage_legacy(
                dir.path(),
                "operator-checkpoint",
                &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE],
            );
            std::env::set_var(key, &pinned);
            let resolved = resolve_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()));
            std::env::remove_var(key);
            assert_eq!(
                resolved.expect("resolves"),
                Some(pinned),
                "{key} must win over the installed snapshot"
            );
        }
    }

    /// sc-17632 review — **a stale `SCENEWORKS_SEEDVR2_DIR` must not make an installed model
    /// unreachable.** That key used to be a download DESTINATION on the video lane, so an empty one
    /// is its normal pre-first-use state, and after this story it is consulted on the IMAGE lane
    /// too — a developer who exported it and never populated it would otherwise start hard-failing
    /// image upscales that the export never affected before.
    ///
    /// End-to-end through the real resolver, not just the pin helper: incomplete pin + a valid
    /// install ⇒ the install wins.
    #[test]
    fn an_incomplete_staging_dir_pin_falls_through_to_the_hf_cache() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = stage_install(dir.path(), &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]);
        // Present but empty — exactly what the pre-sc-17632 video lane created before downloading.
        let stale = dir.path().join("stale-staging-dir");
        std::fs::create_dir_all(&stale).expect("mk stale");
        std::env::set_var("SCENEWORKS_SEEDVR2_DIR", &stale);
        let resolved = resolve_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()));
        std::env::remove_var("SCENEWORKS_SEEDVR2_DIR");

        assert_eq!(
            resolved.expect("an incomplete staging-dir pin must not be an error"),
            Some(snapshot),
            "a stale SCENEWORKS_SEEDVR2_DIR must fall through to the installed copy, not shadow it"
        );
    }

    /// The other half of the split: an incomplete `SCENEWORKS_SEEDVR2_CHECKPOINT` still fails loudly
    /// even with a valid install present (sc-8911). It NAMES a checkpoint, so silently loading a
    /// different one is the worse failure — and the message is actionable, naming the key and the
    /// missing file. Staging a complete install is what makes this test meaningful: it proves the
    /// error is chosen over an available fallback rather than reported for lack of one.
    #[test]
    fn an_incomplete_named_checkpoint_pin_still_errors_over_a_valid_install() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        stage_install(dir.path(), &[SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]);
        let stale = dir.path().join("stale-checkpoint");
        std::fs::create_dir_all(&stale).expect("mk stale");
        std::env::set_var("SCENEWORKS_SEEDVR2_CHECKPOINT", &stale);
        let resolved = resolve_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()));
        std::env::remove_var("SCENEWORKS_SEEDVR2_CHECKPOINT");

        let message = match &resolved {
            Err(WorkerError::InvalidPayload(message)) => message.clone(),
            other => panic!("an incomplete named-checkpoint pin must error, got {other:?}"),
        };
        assert!(
            message.contains("SCENEWORKS_SEEDVR2_CHECKPOINT") && message.contains(SEEDVR2_DIT_FILE),
            "the error must name the key and what is missing, got {message}"
        );
    }

    /// AC7 / S10: absent weights produce an actionable install error, not a download.
    #[test]
    fn errors_actionably_when_nothing_is_installed() {
        let _env = isolate();
        let dir = tempfile::tempdir().expect("tempdir");
        let error = require_seedvr2_checkpoint_dir(&settings_at(dir.path().to_path_buf()))
            .expect_err("nothing is installed");
        let message = format!("{error:?}");
        assert!(
            message.contains("Model Manager"),
            "the error must tell the user how to fix it, got {message}"
        );
        assert!(matches!(error, WorkerError::InvalidPayload(_)), "{error:?}");
    }
}

/// sc-17632, epic 17625 rule 6 — the `seedvr2_upscaler` catalog entry must declare byte-for-byte
/// the repo, revision and filenames [`resolve_seedvr2_checkpoint_dir`] reads.
///
/// `SEEDVR2_REVISION` is a 40-hex pin, so the resolver takes `resolve_hf_component_file`'s
/// `huggingface_pinned_snapshot_dir` branch and reads `snapshots/<sha>/` — that EXACT sha, with no
/// fall-through to `refs/main`. A manifest that installed a different revision would therefore
/// install weights the loader cannot see: the entry would report "installed" while every SeedVR2
/// job failed with "not installed". Reading the pin out of the embedded catalog (rather than
/// mirroring it here) means a bump on either side has to be a bump on both.
#[test]
fn manifest_declares_the_same_seedvr2_pin_the_loader_reads() {
    let pin = crate::manifest_pins::builtin_model_pin("seedvr2_upscaler");
    assert_eq!(
        pin.repo, SEEDVR2_REPO,
        "the catalog entry must declare the repo the loader resolves"
    );
    assert_eq!(
        pin.revision, SEEDVR2_REVISION,
        "the catalog entry must declare the EXACT pinned snapshot the loader reads"
    );
    let mut declared = pin.files.clone();
    declared.sort();
    let mut loaded = vec![SEEDVR2_DIT_FILE.to_owned(), SEEDVR2_VAE_FILE.to_owned()];
    loaded.sort();
    assert_eq!(
        declared, loaded,
        "the catalog entry must declare exactly the DiT + VAE filenames the engine loads — the \
         upstream names, unrenamed"
    );
}
