
#[test]
fn now_matches_python_second_precision() {
    let value = now_rfc3339();

    assert!(value.ends_with('Z'));
    assert!(!value.trim_end_matches('Z').contains('.'));
}

#[test]
fn ffmpeg_helper_shapes_match_python_timeline_exporter() {
    assert_eq!(output_dimensions("16:9", 720), (1280, 720));
    assert_eq!(output_dimensions("9:16", 720), (720, 1280));
    assert_eq!(output_dimensions("1:1", 721), (722, 722));

    let concat = concat_file_contents(
        [
            PathBuf::from(r"C:\renders\clip one's.mp4"),
            PathBuf::from("nested/two.mp4"),
        ]
        .iter(),
    );
    assert!(concat.contains("C:/renders/clip one'\\''s.mp4"));
    assert!(concat.contains("file 'nested/two.mp4'"));

    let asset_id = fresh_asset_id();
    assert!(asset_id.starts_with("asset_"));
    assert_eq!(asset_id.len(), "asset_".len() + 32);
    assert!(asset_id["asset_".len()..]
        .chars()
        .all(|character| character.is_ascii_hexdigit()));
}

#[test]
fn plan_segments_inserts_gaps_and_totals_duration() {
    let items = vec![
        json!({"assetId": "a", "timelineStart": 1.0, "timelineEnd": 3.0}),
        json!({"assetId": "b", "timelineStart": 3.0, "timelineEnd": 5.0}),
        json!({"assetId": "c", "timelineStart": 6.5, "timelineEnd": 8.0}),
    ];

    let (plan, duration) = plan_segments(&items).expect("plan succeeds");

    assert_eq!(plan.len(), 3);
    // Leading hole before the first item becomes a black gap.
    assert_eq!(plan[0].leading_gap, Some(1.0));
    // Abutting items leave no gap.
    assert_eq!(plan[1].leading_gap, None);
    // Interior hole between items becomes a gap of the missing span.
    assert_eq!(plan[2].leading_gap, Some(1.5));
    // Total duration is the running max of item ends.
    assert_eq!(duration, 8.0);
}

#[test]
fn plan_segments_carries_item_transitions() {
    let items = vec![
        json!({
            "assetId": "a",
            "timelineStart": 0.0,
            "timelineEnd": 2.0,
            "transitionIn": {"type": "crossfade", "duration": 0.8}
        }),
        json!({"assetId": "b", "timelineStart": 2.0, "timelineEnd": 4.0}),
    ];

    let (plan, _) = plan_segments(&items).expect("plan succeeds");

    assert_eq!(plan[0].transition.as_deref(), Some("crossfade"));
    assert_eq!(plan[0].transition_duration, 0.8);
    // Missing transitionIn falls back to the default transition duration.
    assert_eq!(plan[1].transition, None);
    assert_eq!(
        plan[1].transition_duration,
        DEFAULT_TRANSITION_DURATION_SECONDS
    );
}

#[test]
fn plan_segments_rejects_nonpositive_item_span() {
    let items = vec![json!({"assetId": "a", "timelineStart": 2.0, "timelineEnd": 2.0})];

    let error = plan_segments(&items).expect_err("zero-length span rejects");

    assert!(matches!(error, WorkerError::InvalidPayload(_)));
    assert!(error.to_string().contains("timelineEnd must be greater"));
}

#[test]
fn person_detection_jitter_uses_python_sha256_bytes() {
    let detections = candidate_people(1280, 720, "asset_source_clip", 1.25);

    assert_eq!(detections[0]["box"]["x"].as_f64(), Some(0.338));
    assert_eq!(detections[1]["box"]["x"].as_f64(), Some(0.579));
    assert_eq!(detections[2]["box"]["x"].as_f64(), Some(0.134));
}

#[test]
fn missing_crossfade_duration_defaults_to_python_mux_duration() {
    let missing = json!(null);
    assert_eq!(
        value_f64(&missing, DEFAULT_TRANSITION_DURATION_SECONDS),
        0.5
    );
    assert_eq!(crossfade_duration(0.5), 0.5);
    assert_eq!(crossfade_duration(0.0), 0.1);
    assert_eq!(crossfade_duration(2.0), 1.5);
}

#[test]
fn path_and_error_helpers_are_bounded_and_defensive() {
    let temp = tempdir().expect("tempdir creates");
    let error = safe_project_path(temp.path(), "").expect_err("empty relative path rejects");
    assert!(error
        .to_string()
        .contains("Project-relative path is required"));

    // sc-4278 / F-MLXW-14: load_reference_image and resolve_clip_media_path route
    // sidecar-sourced media paths through safe_project_path, so a traversal or
    // absolute path (from a poisoned, user-editable sidecar) must be rejected
    // rather than escaping the project.
    for unsafe_rel in ["../../etc/passwd", "assets/../../secret.png", "/etc/passwd"] {
        let error = safe_project_path(temp.path(), unsafe_rel)
            .expect_err("traversal/absolute path rejects");
        assert!(
            error.to_string().contains("Unsafe project-relative path"),
            "{unsafe_rel} should be rejected as unsafe, got {error}"
        );
    }
    // A normal project-relative media path still resolves under the project root.
    let safe = safe_project_path(temp.path(), "assets/images/x.png").expect("safe path resolves");
    assert!(safe.starts_with(temp.path()));
    assert!(safe.ends_with("assets/images/x.png"));

    let noisy = (0..100)
        .map(|index| format!("line {index} caf\u{e9}"))
        .collect::<Vec<_>>()
        .join("\n");
    let tail = bounded_tail(&noisy, 10, 37);

    assert!(tail.contains("caf\u{e9}"));
    assert!(!tail.contains("line 1 "));
}

#[test]
fn model_destinations_are_constrained_to_data_models() {
    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = temp.path().to_path_buf();
    // sc-9812: the confinement helpers canonicalize the deepest existing ancestor of a
    // (possibly not-yet-created) target, so the returned path is expressed via the
    // canonical data-dir root (on macOS `/var` -> `/private/var`). Build the expected
    // prefix from the canonical tempdir so `starts_with` matches.
    let canonical_root = temp.path().canonicalize().expect("tempdir canonicalizes");
    let models_root = super::normalize_absolute_path(&canonical_root.join("models"))
        .expect("models root normalizes");
    let fallback = temp.path().join("models").join("fallback");

    // model_download/model_import: a targetDir under data/models is accepted.
    let mut payload = JsonObject::new();
    payload.insert(
        "targetDir".to_owned(),
        Value::String(
            temp.path()
                .join("models")
                .join("z_image_turbo")
                .display()
                .to_string(),
        ),
    );
    let resolved = resolve_model_import_target(&settings, &payload, fallback.clone())
        .expect("destination under data/models is accepted");
    assert!(resolved.starts_with(&models_root));

    // No targetDir falls back to the supplied (contained) default.
    let resolved_fallback =
        resolve_model_import_target(&settings, &JsonObject::new(), fallback.clone())
            .expect("fallback under data/models is accepted");
    assert!(resolved_fallback.starts_with(&models_root));

    // A targetDir outside data/models is rejected (arbitrary write blocked).
    let mut escape = JsonObject::new();
    escape.insert(
        "targetDir".to_owned(),
        Value::String(
            temp.path()
                .join("ssh")
                .join("authorized_keys")
                .display()
                .to_string(),
        ),
    );
    let error = resolve_model_import_target(&settings, &escape, fallback)
        .expect_err("destination outside data/models is rejected");
    assert!(error.to_string().contains("data/models"));

    // model_convert: outputDir under data/models is accepted, traversal is rejected.
    let ok = resolve_model_convert_output(
        &settings,
        &temp
            .path()
            .join("models")
            .join("mlx")
            .join("wan")
            .display()
            .to_string(),
    )
    .expect("convert output under data/models is accepted");
    assert!(ok.starts_with(&models_root));

    let traversal = temp
        .path()
        .join("models")
        .join("..")
        .join("escape")
        .display()
        .to_string();
    let convert_error = resolve_model_convert_output(&settings, &traversal)
        .expect_err("convert output escaping data/models is rejected");
    assert!(convert_error.to_string().contains("data/models"));
}

#[cfg(unix)]
#[test]
fn lora_paths_resolve_symlinks_before_root_check() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let lora_dir = data_dir.join("loras");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    let safe = lora_dir.join("safe.safetensors");
    std::fs::write(&safe, b"safe").expect("safe lora writes");
    let normalized =
        super::normalize_app_managed_lora_path(&settings, &safe).expect("safe lora accepted");
    assert_eq!(
        normalized,
        safe.canonicalize().expect("safe lora canonicalizes")
    );

    let outside = outside_dir.join("escape.safetensors");
    std::fs::write(&outside, b"outside").expect("outside lora writes");
    let link = lora_dir.join("escape-link.safetensors");
    std::os::unix::fs::symlink(&outside, &link).expect("symlink creates");

    let error = super::normalize_app_managed_lora_path(&settings, &link)
        .expect_err("symlink target outside managed roots rejects");
    assert!(error.to_string().contains("LoRA path must be inside"));
}

/// epic 10451 / sc-10452: an operator-configured external root (a ComfyUI `models/`
/// tree) is readable for adapters, in place — no copy into `<data>/loras`. The same
/// path is rejected when no root is configured, which is the default and therefore
/// the behaviour every existing install keeps.
#[test]
fn lora_paths_admit_operator_configured_external_roots() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(data_dir.join("loras")).expect("lora dir creates");
    // Mimic the real tree: nested subdirectory under `<root>/loras`.
    let comfy_root = temp.path().join("ComfyUI").join("models");
    let comfy_loras = comfy_root.join("loras").join("Wan");
    std::fs::create_dir_all(&comfy_loras).expect("comfy lora dir creates");
    let adapter = comfy_loras.join("detailz-wan.safetensors");
    std::fs::write(&adapter, b"adapter").expect("adapter writes");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir;

    // Default (no external roots): the ComfyUI file is outside every managed root.
    let error = super::normalize_app_managed_lora_path(&settings, &adapter)
        .expect_err("external path rejects while the feature is off");
    assert!(error.to_string().contains("LoRA path must be inside"));

    // Operator opts in: the very same path now resolves, canonicalized.
    settings.external_model_roots = vec![comfy_root];
    let normalized = super::normalize_app_managed_lora_path(&settings, &adapter)
        .expect("adapter under an external root is accepted");
    assert_eq!(
        normalized,
        adapter.canonicalize().expect("adapter canonicalizes")
    );
}

/// Phase 2 (sc-10668) widened the **base-model** confinement the same way: an external
/// ComfyUI base component (DiT / text-encoder / VAE) is read in place from a configured
/// root. Off by default it rejects; opted in it resolves; a sibling stays rejected.
#[test]
fn model_paths_admit_operator_configured_external_roots() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    let comfy_root = temp.path().join("ComfyUI").join("models");
    let unet = comfy_root.join("unet");
    std::fs::create_dir_all(&unet).expect("unet dir creates");
    let dit = unet.join("z_image_turbo_bf16.safetensors");
    std::fs::write(&dit, b"weights").expect("dit writes");
    let dit_str = dit.display().to_string();

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir;

    // Off by default: the ComfyUI base file is outside every managed root.
    let error = super::normalize_app_managed_model_path(&settings, &dit_str, "Model")
        .expect_err("external base path rejects while the feature is off");
    assert!(error.to_string().contains("must be inside"), "{error}");

    // Operator opts in: the same path now resolves, canonicalized.
    settings.external_model_roots = vec![comfy_root];
    let normalized = super::normalize_app_managed_model_path(&settings, &dit_str, "Model")
        .expect("base component under an external root is accepted");
    assert_eq!(normalized, dit.canonicalize().expect("dit canonicalizes"));

    // A sibling outside the root stays rejected — no arbitrary-file-read widening.
    let secret = temp.path().join("secret.safetensors");
    std::fs::write(&secret, b"secret").expect("secret writes");
    super::normalize_app_managed_model_path(&settings, &secret.display().to_string(), "Model")
        .expect_err("a sibling of the external root is still rejected");
}

/// Configuring an external root must widen the allow-list to *that root only* — a
/// sibling directory stays rejected. Without this, "point at my ComfyUI folder"
/// would quietly become "read anything on the host", which is the arbitrary-file-read
/// primitive the confinement exists to close (epic 4484).
#[test]
fn external_roots_do_not_admit_paths_outside_them() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(data_dir.join("loras")).expect("lora dir creates");
    let comfy_root = temp.path().join("comfy");
    std::fs::create_dir_all(comfy_root.join("loras")).expect("comfy dir creates");
    let secrets = temp.path().join("secrets");
    std::fs::create_dir_all(&secrets).expect("secrets dir creates");
    let stolen = secrets.join("id_rsa");
    std::fs::write(&stolen, b"secret").expect("secret writes");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir;
    settings.external_model_roots = vec![comfy_root];

    let error = super::normalize_app_managed_lora_path(&settings, &stolen)
        .expect_err("a sibling of the external root is still rejected");
    assert!(error.to_string().contains("LoRA path must be inside"));
}

/// A configured root that does not exist (unmounted drive, typo) must not error the
/// whole confinement check — it simply never matches.
#[test]
fn missing_external_root_is_inert_rather_than_fatal() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let lora_dir = data_dir.join("loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    let managed = lora_dir.join("safe.safetensors");
    std::fs::write(&managed, b"safe").expect("safe lora writes");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir;
    settings.external_model_roots = vec![temp.path().join("not-mounted")];

    // The managed path still resolves; the absent root neither errors nor admits.
    super::normalize_app_managed_lora_path(&settings, &managed)
        .expect("a missing external root leaves managed roots working");
}

// sc-8877 / F-075: the write-target confinement helpers must canonicalize before the
// root check so a symlink planted under a managed root can't smuggle a write outside
// via a purely-lexical `starts_with`. Covers all five that used the weaker check.
#[cfg(unix)]
#[test]
fn app_managed_helpers_resolve_symlinks_before_root_check() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let models_dir = data_dir.join("models");
    let loras_dir = data_dir.join("loras");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&models_dir).expect("models dir creates");
    std::fs::create_dir_all(&loras_dir).expect("loras dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    // A symlink under a managed root pointing at an outside dir: a lexical
    // `starts_with` would accept it; canonicalization must make each helper reject it.
    let outside_target = outside_dir.join("escape");
    std::fs::create_dir_all(&outside_target).expect("outside target creates");

    // normalize_app_managed_path (write target confined to data_dir)
    let path_link = data_dir.join("path-escape");
    std::os::unix::fs::symlink(&outside_target, &path_link).expect("symlink creates");
    let err =
        super::normalize_app_managed_path(&settings, &path_link.display().to_string(), "Path")
            .expect_err("symlinked path escape rejects");
    assert!(err.to_string().contains("app-managed directory"));

    // normalize_app_managed_model_path (read source confined to data_dir/hf_cache)
    let model_link = models_dir.join("model-escape");
    std::os::unix::fs::symlink(&outside_target, &model_link).expect("symlink creates");
    let err = super::normalize_app_managed_model_path(
        &settings,
        &model_link.display().to_string(),
        "Model",
    )
    .expect_err("symlinked model escape rejects");
    assert!(err.to_string().contains("app-managed directory"));

    // resolve_lora_import_target (write target confined to data/loras)
    let lora_link = loras_dir.join("lora-escape");
    std::os::unix::fs::symlink(&outside_target, &lora_link).expect("symlink creates");
    let mut lora_payload = JsonObject::new();
    lora_payload.insert(
        "targetDir".to_owned(),
        Value::String(lora_link.display().to_string()),
    );
    let err = super::resolve_lora_import_target(&settings, &lora_payload, loras_dir.clone())
        .expect_err("symlinked lora targetDir escape rejects");
    assert!(err.to_string().contains("data/loras"));

    // resolve_model_import_target (write target confined to data/models)
    let import_link = models_dir.join("import-escape");
    std::os::unix::fs::symlink(&outside_target, &import_link).expect("symlink creates");
    let mut model_payload = JsonObject::new();
    model_payload.insert(
        "targetDir".to_owned(),
        Value::String(import_link.display().to_string()),
    );
    let err = super::resolve_model_import_target(&settings, &model_payload, models_dir.clone())
        .expect_err("symlinked model targetDir escape rejects");
    assert!(err.to_string().contains("data/models"));

    // resolve_model_convert_output (write target confined to data/models)
    let convert_link = models_dir.join("convert-escape");
    std::os::unix::fs::symlink(&outside_target, &convert_link).expect("symlink creates");
    let err = super::resolve_model_convert_output(&settings, &convert_link.display().to_string())
        .expect_err("symlinked convert outputDir escape rejects");
    assert!(err.to_string().contains("data/models"));

    // Sanity: a genuine dir under a managed root still resolves (not over-rejected).
    let real = models_dir.join("real_model");
    std::fs::create_dir_all(&real).expect("real model dir creates");
    super::normalize_app_managed_model_path(&settings, &real.display().to_string(), "Model")
        .expect("a real managed dir is still accepted");
}

// sc-9812 / F-075 follow-up: the sc-8877 fix canonicalizes before the confinement
// check, but `normalize_existing_or_absolute` fell back to purely-*lexical*
// normalization whenever `canonicalize` returned `NotFound` — which fires as soon as
// the *leaf* is absent, even if an *intermediate* directory is a symlink escaping the
// root. So `<data>/loras/evil-symlink/newdir` (evil-symlink -> outside, newdir not yet
// created) resolved lexically and still satisfied `starts_with(<data>)`. The fix
// canonicalizes the deepest existing ancestor (resolving the intermediate symlink)
// before re-appending the missing tail, so the confinement check now catches it.
#[cfg(unix)]
#[test]
fn intermediate_dir_symlink_escape_with_nonexistent_leaf_is_rejected() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let loras_dir = data_dir.join("loras");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&loras_dir).expect("loras dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    // 1. A normal not-yet-existing leaf under the root still resolves and passes.
    let fresh_leaf = loras_dir.join("brand-new-dir");
    let normalized =
        super::normalize_app_managed_path(&settings, &fresh_leaf.display().to_string(), "Path")
            .expect("a not-yet-created leaf under the root is still accepted");
    // The resolved path stays under the (canonicalized) managed root.
    assert!(normalized.starts_with(
        data_dir
            .canonicalize()
            .expect("data dir canonicalizes")
    ));

    // 2. An existing valid path under the root still passes.
    let existing = loras_dir.join("existing-dir");
    std::fs::create_dir_all(&existing).expect("existing dir creates");
    super::normalize_app_managed_path(&settings, &existing.display().to_string(), "Path")
        .expect("an existing dir under the root is still accepted");

    // 3. The exploit: an *intermediate* directory symlink escaping the root, with a
    //    nonexistent leaf beyond it. Pre-fix this passed the lexical `starts_with`.
    let outside_target = outside_dir.join("escape");
    std::fs::create_dir_all(&outside_target).expect("outside target creates");
    let evil_symlink = loras_dir.join("evil-symlink");
    std::os::unix::fs::symlink(&outside_target, &evil_symlink).expect("symlink creates");
    let escaped = evil_symlink.join("newdir"); // newdir does not exist yet
    let err =
        super::normalize_app_managed_path(&settings, &escaped.display().to_string(), "Path")
            .expect_err("intermediate-symlink escape with a nonexistent leaf rejects");
    assert!(err.to_string().contains("app-managed directory"));

    // 4. Regression guard for sc-8877: a *leaf* symlink escape is still rejected.
    let leaf_symlink = loras_dir.join("leaf-escape");
    std::os::unix::fs::symlink(&outside_target, &leaf_symlink).expect("symlink creates");
    let err = super::normalize_app_managed_path(
        &settings,
        &leaf_symlink.display().to_string(),
        "Path",
    )
    .expect_err("leaf-symlink escape still rejects");
    assert!(err.to_string().contains("app-managed directory"));
}

// sc-8821 / F-019: payload-supplied weight filenames (`advanced.controlWeights.filename`,
// `advanced.pidCheckpoint.filename`) are joined under a resolved HF snapshot / app-cache
// dir, so they must be a single plain path component — traversal, absolute paths, and
// sub-paths are rejected before any join.
#[test]
fn weight_filenames_are_confined_to_plain_components() {
    // Plain filenames pass through (trimmed).
    assert_eq!(
        super::safe_weight_filename("model.safetensors", "advanced.controlWeights.filename")
            .expect("plain filename accepted"),
        "model.safetensors"
    );
    assert_eq!(
        super::safe_weight_filename("  model.safetensors  ", "advanced.controlWeights.filename")
            .expect("surrounding whitespace trimmed"),
        "model.safetensors"
    );

    for unsafe_name in [
        "../../etc/hosts",
        "..",
        ".",
        "",
        "/etc/hosts",
        "sub/dir.safetensors",
        "..\\..\\secrets.safetensors",
        "sub\\dir.safetensors",
        "model.safetensors/",
    ] {
        let error = super::safe_weight_filename(unsafe_name, "advanced.controlWeights.filename")
            .expect_err("non-plain filename rejected");
        assert!(
            error
                .to_string()
                .contains("advanced.controlWeights.filename"),
            "error names the offending field for {unsafe_name:?}: {error}"
        );
    }
}

// sc-8803 / F-002: LoRA/model import *source* paths are client-supplied over the
// unauthenticated jobs API; the worker must confine them before copying (or, for
// uploads, moving) the file into an app-listable directory.
#[test]
fn import_source_paths_are_confined_to_app_managed_roots() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let staged_dir = data_dir
        .join("cache")
        .join("lora-uploads")
        .join("upload-abc");
    let loras_dir = data_dir.join("loras").join("existing");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&staged_dir).expect("staged dir creates");
    std::fs::create_dir_all(&loras_dir).expect("loras dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    // Uploaded (move-mode) sources are accepted only from the staged-upload cache.
    let staged_file = staged_dir.join("lora.safetensors");
    std::fs::write(&staged_file, b"staged").expect("staged file writes");
    let mut uploaded = JsonObject::new();
    uploaded.insert("uploadedSourcePath".to_owned(), Value::Bool(true));
    let resolved = super::resolve_import_source_path(
        &settings,
        &uploaded,
        &staged_file.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect("staged upload source accepted");
    assert!(resolved.ends_with("lora.safetensors"));

    // Uploaded flag with a source elsewhere in data_dir is rejected: move mode
    // would otherwise delete arbitrary app-managed files.
    let installed_file = loras_dir.join("installed.safetensors");
    std::fs::write(&installed_file, b"installed").expect("installed file writes");
    let error = super::resolve_import_source_path(
        &settings,
        &uploaded,
        &installed_file.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("uploaded source outside the staged-upload cache rejects");
    assert!(error
        .to_string()
        .contains("LoRA import sourcePath must be inside"));

    // Copy-mode sources under data_dir are accepted (re-import of an installed file).
    let copy_payload = JsonObject::new();
    super::resolve_import_source_path(
        &settings,
        &copy_payload,
        &installed_file.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect("data_dir source accepted for copy-mode import");

    // The model-import upload cache confines the same way.
    let model_staged = data_dir
        .join("cache")
        .join("model-uploads")
        .join("upload-def");
    std::fs::create_dir_all(&model_staged).expect("model staged dir creates");
    let model_file = model_staged.join("model.safetensors");
    std::fs::write(&model_file, b"model").expect("model file writes");
    super::resolve_import_source_path(
        &settings,
        &uploaded,
        &model_file.display().to_string(),
        "model-uploads",
        "Model import sourcePath",
    )
    .expect("staged model upload accepted");

    // An absolute path outside data_dir (the exfiltration primitive) is rejected
    // in both modes.
    let secret = outside_dir.join("id_rsa");
    std::fs::write(&secret, b"secret").expect("secret writes");
    for payload in [&uploaded, &copy_payload] {
        let error = super::resolve_import_source_path(
            &settings,
            payload,
            &secret.display().to_string(),
            "lora-uploads",
            "LoRA import sourcePath",
        )
        .expect_err("host path outside app-managed roots rejects");
        assert!(error
            .to_string()
            .contains("LoRA import sourcePath must be inside"));
    }

    // A `..` traversal that starts inside data_dir but escapes is rejected.
    let traversal = data_dir
        .join("loras")
        .join("..")
        .join("..")
        .join("outside")
        .join("id_rsa");
    super::resolve_import_source_path(
        &settings,
        &copy_payload,
        &traversal.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("traversal escape rejects");

    // An empty source path is rejected.
    super::resolve_import_source_path(
        &settings,
        &copy_payload,
        "  ",
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("empty source path rejects");
}

// Symlinks resolve before the root check, so a link planted inside data_dir cannot
// smuggle an outside file through the import copy.
#[cfg(unix)]
#[test]
fn import_source_symlink_escape_is_rejected() {
    let temp = tempdir().expect("tempdir creates");
    let data_dir = temp.path().join("data");
    let loras_dir = data_dir.join("loras");
    let outside_dir = temp.path().join("outside");
    std::fs::create_dir_all(&loras_dir).expect("loras dir creates");
    std::fs::create_dir_all(&outside_dir).expect("outside dir creates");

    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = data_dir.clone();

    let outside_file = outside_dir.join("escape.safetensors");
    std::fs::write(&outside_file, b"outside").expect("outside file writes");
    let link = loras_dir.join("escape-link.safetensors");
    std::os::unix::fs::symlink(&outside_file, &link).expect("symlink creates");

    let error = super::resolve_import_source_path(
        &settings,
        &JsonObject::new(),
        &link.display().to_string(),
        "lora-uploads",
        "LoRA import sourcePath",
    )
    .expect_err("symlink target outside managed roots rejects");
    assert!(error
        .to_string()
        .contains("LoRA import sourcePath must be inside"));
}

// sc-8898 / F-096: a missing import source now surfaces the friendly "LoRA source
// not found" message. Previously `canonicalize()` failed NotFound first and the
// `!exists()` branch that built this message was dead, so the user only saw the
// raw OS error.
#[tokio::test]
async fn missing_lora_import_source_reports_friendly_not_found() {
    let temp = tempdir().expect("tempdir creates");
    let missing = temp.path().join("does-not-exist.safetensors");
    let target_dir = temp.path().join("target");

    let error = import_lora_source_path(&missing, &target_dir, false)
        .await
        .expect_err("missing source errors");

    match error {
        WorkerError::Io(io_error) => {
            assert_eq!(io_error.kind(), std::io::ErrorKind::NotFound);
            assert!(
                io_error.to_string().contains("LoRA source not found"),
                "unexpected message: {io_error}"
            );
        }
        other => panic!("expected NotFound Io error, got {other:?}"),
    }
    // The target dir is not created for a missing source (the copy never runs).
    assert!(!target_dir.exists());
}

#[tokio::test]
async fn ffmpeg_runner_surfaces_bounded_stderr_from_failing_process() {
    let args = if cfg!(windows) {
        let command = (1..=30)
            .map(|index| format!("echo ffmpeg-line-{index} 1>&2"))
            .collect::<Vec<_>>()
            .join(" & ");
        vec![
            "cmd".to_owned(),
            "/C".to_owned(),
            format!("{command} & exit /B 7"),
        ]
    } else {
        vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "for i in $(seq 1 30); do echo ffmpeg-line-$i >&2; done; exit 7".to_owned(),
        ]
    };

    let error = run_ffmpeg(args, None)
        .await
        .expect_err("non-zero process returns an error");

    match error {
        WorkerError::Engine(message) => {
            assert!(message.contains("ffmpeg-line-30"));
            assert!(!message.contains("ffmpeg-line-1"));
            assert!(message.len() <= 2000);
        }
        other => panic!("expected Engine, got {other:?}"),
    }
}
