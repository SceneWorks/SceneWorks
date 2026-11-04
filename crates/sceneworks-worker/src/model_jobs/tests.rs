#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// sc-6072: `safetensors_file_is_unusable` selects exactly the cases where the
    /// file on disk is itself bad (so the import path discards it) versus a transient
    /// read failure (kept for a retry).
    #[test]
    fn safetensors_file_is_unusable_classifies_variants() {
        assert!(!safetensors_file_is_unusable(&SafetensorsHeaderError::Io(
            std::io::Error::other("transient")
        )));
        assert!(safetensors_file_is_unusable(
            &SafetensorsHeaderError::InvalidHeader
        ));
        assert!(safetensors_file_is_unusable(
            &SafetensorsHeaderError::IncompleteData {
                declared: 1024,
                actual: 16,
            }
        ));
    }

    /// sc-6072: a truncated `.safetensors` in the import target dir is rejected by the
    /// import-time inspector with an actionable, file-incompleteness message.
    #[test]
    fn inspect_adapter_in_dir_rejects_truncated_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let header = json!({
            "__metadata__": { "format": "pt" },
            "lora.down.weight": { "dtype": "F16", "shape": [16, 16], "data_offsets": [0, 512] },
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize");
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        // Declares 512 bytes of tensor data but writes none — truncated download.
        std::fs::write(dir.path().join("lora.safetensors"), buffer).expect("write");

        let error =
            inspect_adapter_in_dir(dir.path()).expect_err("a truncated file must not be accepted");
        assert!(
            matches!(error, SafetensorsHeaderError::IncompleteData { .. }),
            "expected IncompleteData, got {error:?}"
        );
        assert!(safetensors_file_is_unusable(&error));
        let message = lora_family_detection_error(&error).to_ascii_lowercase();
        assert!(
            message.contains("incomplete") && message.contains("re-import"),
            "message should be actionable, got: {message}"
        );
    }

    /// Resolve a HuggingFace cache snapshot dir for `models--<dir>` (test helper).
    fn hf_snapshot(model_dir: &str) -> PathBuf {
        let home = std::env::var("HOME").expect("HOME");
        std::fs::read_dir(
            Path::new(&home).join(format!(".cache/huggingface/hub/{model_dir}/snapshots")),
        )
        .expect("HF cache snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a snapshot dir")
    }

    /// Real-weights smoke for the native Rust/MLX FLUX.2-klein true_v2 converter
    /// (sc-3136 consumer cutover, sc-3032): runs the actual `convert_flux2_klein_diffusers`
    /// wrapper used by `run_model_convert_job` on the cached wikeeyang bf16 single-file +
    /// base klein-9B, and asserts the assembled diffusers dir is complete (real transformer
    /// weights + config + model_index) with the borrowed components as resolvable symlinks.
    /// Mirrors mlx-gen's `convert_real_weights` but exercises the SceneWorks call site.
    /// Needs both repos in the HF cache (writes ~18 GB to the temp dir). Run with:
    ///   cargo test -p sceneworks-worker --lib -- --ignored flux2_true_v2_rust_convert
    #[test]
    #[ignore]
    fn flux2_true_v2_rust_convert_real_weights() {
        let source = hf_snapshot("models--wikeeyang--Flux2-Klein-9B-True-V2")
            .join("Flux2-Klein-9B-True-v2-bf16.safetensors");
        let base = hf_snapshot("models--black-forest-labs--FLUX.2-klein-9B");
        assert!(
            source.is_file(),
            "missing wikeeyang bf16 single-file: {}",
            source.display()
        );
        assert!(
            base.join("transformer").is_dir(),
            "missing base klein transformer: {}",
            base.display()
        );

        let out_guard = tempfile::Builder::new()
            .prefix("sw_true_v2_convert_")
            .tempdir()
            .expect("temp dir");
        let out = out_guard.path();

        convert_flux2_klein_diffusers(&source, &base, out).expect("native true_v2 convert");

        // Transformer weights + config are written as real files (the remapped, base-validated
        // diffusers transformer); model_index.json is copied.
        assert!(out
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file());
        assert!(out.join("transformer/config.json").is_file());
        assert!(out.join("model_index.json").is_file());
        // vae / text_encoder / tokenizer / scheduler are absolute symlinks borrowed from the
        // base install; `.exists()` follows the link, so a broken symlink fails here.
        for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
            assert!(
                out.join(sub).exists(),
                "borrowed `{sub}` missing or broken symlink in {}",
                out.display()
            );
        }
    }

    /// Real-weights smoke for the native Rust/MLX LTX-2.3 converter (sc-3224 engine + sc-3240
    /// cutover) — the LTX path was never routed through the Rust job before. Runs the actual
    /// `convert_ltx_native` used by `run_model_convert_job` on the cached eros single-file + the base
    /// LTX-2.3 upscaler, and asserts the split MLX dir is complete, including the
    /// `upsampler.safetensors` the loader hard-requires (merged from the base repo) and the
    /// `split_model.json` Q4 quant manifest. Needs both repos in the HF cache. Run with:
    ///   cargo test -p sceneworks-worker --lib -- --ignored ltx_eros_rust_convert
    #[test]
    #[ignore]
    fn ltx_eros_rust_convert_real_weights() {
        let source =
            hf_snapshot("models--TenStrip--LTX2.3-10Eros").join("10Eros_v1.3_bf16.safetensors");
        let upscaler_dir = hf_snapshot("models--Lightricks--LTX-2.3");
        assert!(
            source.is_file(),
            "missing eros single-file checkpoint: {}",
            source.display()
        );
        assert!(
            upscaler_dir.join(LTX_SPATIAL_UPSCALER_FILE).is_file(),
            "missing base LTX-2.3 spatial upscaler: {}",
            upscaler_dir.display()
        );

        let out_guard = tempfile::Builder::new()
            .prefix("sw_ltx_eros_convert_")
            .tempdir()
            .expect("temp dir");
        let out = out_guard.path();

        convert_ltx_native(&source, &upscaler_dir, out, 4).expect("native LTX-2.3 eros convert");

        for file in [
            "transformer.safetensors",
            "connector.safetensors",
            // The loader-required latent upsampler, merged from the base LTX-2.3 repo (sc-3240).
            "upsampler.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "config.json",
            "embedded_config.json",
            // The Q4 quant geometry the loader reads back (sc-2686).
            "split_model.json",
        ] {
            assert!(
                out.join(file).is_file(),
                "converted LTX dir missing `{file}` in {}",
                out.display()
            );
        }
    }
}

// Cross-platform (not macOS-gated like the module above): exercises the Windows
// symlink-repair logic, which is compiled under `test` on every platform.
#[cfg(test)]
mod hardlink_tests {
    use super::*;

    /// The HF-cache symlink-traversal fix: a `snapshots/<rev>/<sub>/file ->
    /// ../blobs/<etag>` symlink is replaced in place by a non-reparse entry (hardlink,
    /// or a copy fallback) pointing at the same bytes, and the blob is left untouched.
    /// Recurses into component sub-dirs. Skips when the platform won't let the test
    /// create a symlink (e.g. Windows without Developer Mode).
    #[test]
    fn materialize_snapshot_hardlinks_replaces_symlinks_with_real_files() {
        let root = tempfile::tempdir().expect("tempdir");
        let blobs = root.path().join("blobs");
        std::fs::create_dir_all(&blobs).expect("blobs dir");
        let blob = blobs.join("deadbeef");
        std::fs::write(&blob, b"tokenizer-bytes").expect("write blob");

        let snapshot = root.path().join("snapshots").join("rev");
        let tokenizer_dir = snapshot.join("tokenizer");
        std::fs::create_dir_all(&tokenizer_dir).expect("tokenizer dir");
        let link = tokenizer_dir.join("tokenizer.json");
        // Relative, like huggingface_hub: snapshots/rev/tokenizer/ -> ../../../blobs.
        let rel_target: PathBuf = ["..", "..", "..", "blobs", "deadbeef"].iter().collect();

        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&rel_target, &link).is_ok();
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_file(&rel_target, &link).is_ok();
        #[cfg(not(any(unix, windows)))]
        let made = false;
        if !made {
            return; // no privilege to create a symlink — nothing to convert
        }
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "precondition: entry is a symlink before repair"
        );

        materialize_snapshot_hardlinks(&snapshot);

        let meta = std::fs::symlink_metadata(&link).expect("metadata after repair");
        assert!(
            !meta.file_type().is_symlink(),
            "the symlink must be replaced by a non-reparse entry"
        );
        assert_eq!(
            std::fs::read(&link).expect("read repaired entry"),
            b"tokenizer-bytes"
        );
        assert_eq!(
            std::fs::read(&blob).expect("read blob"),
            b"tokenizer-bytes",
            "the blob itself must be untouched"
        );
    }
}

#[cfg(test)]
mod flux2_convert_tests {
    use super::*;

    fn seed_tier(root: &Path, tier: &str) {
        for comp in ["tokenizer", "text_encoder", "vae"] {
            std::fs::create_dir_all(root.join(tier).join(comp)).unwrap();
        }
    }

    /// sc-14978: the FLUX.2-klein convert borrows VAE/text-encoder/tokenizer from a quant-tier subdir of
    /// the installed re-host (they are dense and identical across tiers). The resolver prefers the
    /// manifest tier, falls back to any present tier, and rejects a torn base — so pointing
    /// `convertBaseRepo` at the ungated re-host (whose components live under `<tier>/`, not the snapshot
    /// root) resolves instead of failing deep with `tokenizer: No such file or directory`.
    #[test]
    fn base_dir_prefers_named_tier_then_falls_back_then_rejects_torn() {
        // Preferred tier present → used.
        let tmp = tempfile::tempdir().unwrap();
        seed_tier(tmp.path(), "bf16");
        assert_eq!(
            resolve_flux2_klein_base_dir(tmp.path(), Some("bf16")),
            Some(tmp.path().join("bf16")),
        );

        // Preferred tier absent but another tier present → fall back to it (borrow is tier-invariant).
        let tmp2 = tempfile::tempdir().unwrap();
        seed_tier(tmp2.path(), "q4");
        assert_eq!(
            resolve_flux2_klein_base_dir(tmp2.path(), Some("bf16")),
            Some(tmp2.path().join("q4")),
        );

        // No tier holds the borrowed components → None (a torn base; the caller surfaces a re-download).
        let tmp3 = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp3.path().join("bf16").join("transformer")).unwrap();
        assert_eq!(
            resolve_flux2_klein_base_dir(tmp3.path(), Some("bf16")),
            None
        );
    }

    /// No `convertBaseSubdir` → a plain root-layout diffusers base borrows from the snapshot root itself.
    #[test]
    fn base_dir_without_subdir_uses_root_layout() {
        let tmp = tempfile::tempdir().unwrap();
        for comp in ["tokenizer", "text_encoder", "vae"] {
            std::fs::create_dir_all(tmp.path().join(comp)).unwrap();
        }
        assert_eq!(
            resolve_flux2_klein_base_dir(tmp.path(), None),
            Some(tmp.path().to_path_buf()),
        );
        // Root missing the borrowed components → None.
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(resolve_flux2_klein_base_dir(empty.path(), None), None);
    }
}

#[cfg(test)]
mod supra2_convert_tests {
    use super::*;
    use serde_json::json;

    fn settings_at(data_dir: PathBuf) -> Settings {
        Settings {
            api_url: "http://127.0.0.1:0".to_owned(),
            config_dir: PathBuf::from("config"),
            worker_id: "sc-24213".to_owned(),
            is_child_worker: true,
            heartbeat_seconds: 5,
            huggingface_base_url: "http://127.0.0.1:0".to_owned(),
            max_lora_url_bytes: 0,
            max_model_url_bytes: 0,
            backend_mlx_enabled: true,
            ..crate::Settings::for_test(data_dir)
        }
    }

    /// The payload `POST /models/supra2_img/convert` builds from the shipped catalog row.
    fn supra2_job(source_file: &str) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job-supra2-convert",
            "type": "model_convert",
            "status": "running",
            "projectId": null,
            "projectName": null,
            "payload": {
                "modelId": "supra2_img",
                "sourceRepo": "SupraLabs/Supra2-IMG",
                "converter": "supra2_img",
                "sourceFile": source_file,
                "outputDir": "/unused"
            },
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0.0,
            "stage": "queued",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "elapsedSeconds": null,
            "attempts": 0,
            "sourceJobId": null,
            "duplicateOfJobId": null,
            "cancelRequested": false,
            "createdAt": "2026-10-20T00:00:00Z",
            "updatedAt": "2026-10-20T00:00:00Z",
            "startedAt": null,
            "finishedAt": null
        }))
        .expect("valid convert job snapshot")
    }

    /// Stage `files` (placeholder bytes) in `repo`'s cached snapshot at exactly `revision`.
    fn stage_snapshot(data_dir: &Path, repo: &str, revision: &str, files: &[&str]) -> PathBuf {
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).expect("create snapshot dir");
        for file in files {
            std::fs::write(snapshot.join(file), b"weights").expect("write staged file");
        }
        snapshot
    }

    async fn plan(
        settings: &Settings,
        checkpoint_dir: &Path,
        source_file: &str,
    ) -> Result<ConvertPlan, ConvertPlanError> {
        let api = ApiClient::new(settings);
        resolve_convert_plan(
            &api,
            settings,
            &supra2_job(source_file),
            checkpoint_dir,
            None,
        )
        .await
        .expect("resolver runs without an infra error")
    }

    /// The shipped catalog row and the converter agree on the one pinned source file, and the
    /// component pins the converter verifies are the exact co-requisite rows the install downloads
    /// (a drift here would verify one revision and load another).
    #[test]
    fn shipped_catalog_row_matches_the_converter_and_component_pins() {
        let raw = include_str!("../../../../config/manifests/builtin.models.jsonc");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin manifest parses");
        let entry = manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"] == "supra2_img")
            .expect("supra2_img is in the shipped catalog");
        let mlx = &entry["mlx"];
        assert_eq!(mlx["requiresConversion"], true);
        assert_eq!(mlx["converter"], "supra2_img");
        assert_eq!(mlx["convertSourceFile"], SUPRA2_SOURCE_FILE);
        let source = gen_core_checkpoint::supra2::SUPRA2_SOURCE;
        assert_eq!(mlx["convertSourceRepo"], source.repo);
        let downloads = entry["downloads"].as_array().expect("downloads");
        let primary = downloads
            .iter()
            .find(|download| download["coRequisite"] != true)
            .expect("primary download");
        assert_eq!(primary["repo"], source.repo);
        assert_eq!(primary["revision"], source.revision);
        assert_eq!(primary["files"], json!([SUPRA2_SOURCE_FILE]));
        assert_eq!(primary["expectedSizeBytes"], source.files[0].bytes);
        assert_eq!(primary["expectedSha256"], source.files[0].sha256);
        // sc-24216: the per-component licence the Model Manager lists is the engine pin's own, and
        // the prompt limit it states is the engine's Flan-T5 context.
        assert_eq!(primary["license"], source.license);
        assert_eq!(
            entry["image"]["maxPromptTokens"],
            gen_core_checkpoint::supra2::MAX_PROMPT_TOKENS
        );
        for (component, pin) in [
            ("text_encoder", gen_core_checkpoint::supra2::FLAN_T5_BASE),
            ("vae", gen_core_checkpoint::supra2::SD_VAE),
        ] {
            let row = downloads
                .iter()
                .find(|download| download["componentId"] == component)
                .unwrap_or_else(|| panic!("{component} co-requisite row"));
            assert_eq!(row["license"], pin.license, "{component}");
        }
        for (component, pin) in [
            ("text_encoder", gen_core_checkpoint::supra2::FLAN_T5_BASE),
            ("vae", gen_core_checkpoint::supra2::SD_VAE),
        ] {
            let row = downloads
                .iter()
                .find(|download| download["componentId"] == component)
                .unwrap_or_else(|| panic!("{component} co-requisite row"));
            assert_eq!(row["coRequisite"], true, "{component}");
            assert_eq!(row["repo"], pin.repo, "{component}");
            assert_eq!(row["revision"], pin.revision, "{component}");
            let files: Vec<&str> = pin.files.iter().map(|file| file.path).collect();
            assert_eq!(row["files"], json!(files), "{component}");
            assert_eq!(
                row["estimatedSizeBytes"],
                pin.download_bytes(),
                "{component}"
            );
        }
    }

    #[tokio::test]
    async fn supra2_plan_refuses_a_foreign_source_file() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let settings = settings_at(tmp.path().join("data"));
        let error = plan(&settings, tmp.path(), "model.safetensors")
            .await
            .err()
            .expect("a foreign source file is refused");
        assert_eq!(error.message, "Unknown Supra2-IMG source checkpoint.");
    }

    #[tokio::test]
    async fn supra2_plan_refuses_a_missing_checkpoint() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let settings = settings_at(tmp.path().join("data"));
        let error = plan(&settings, tmp.path(), SUPRA2_SOURCE_FILE)
            .await
            .err()
            .expect("a missing checkpoint is refused");
        assert_eq!(error.message, "Supra2-IMG source checkpoint is missing.");
    }

    /// The co-requisites are resolved at their exact pinned revision, never `refs/main`: a snapshot
    /// of another revision does not satisfy the convert, and the refusal names the install to make.
    #[tokio::test]
    async fn supra2_plan_resolves_only_the_pinned_component_snapshots() {
        let _env = crate::test_env::isolate_hf_cache();
        let tmp = tempfile::tempdir().expect("temp dir");
        let data_dir = tmp.path().join("data");
        let settings = settings_at(data_dir.clone());
        let checkpoint = tmp.path().join("checkpoint");
        std::fs::create_dir_all(&checkpoint).expect("checkpoint dir");
        std::fs::write(checkpoint.join(SUPRA2_SOURCE_FILE), b"pt").expect("checkpoint");
        let t5 = gen_core_checkpoint::supra2::FLAN_T5_BASE;
        let vae = gen_core_checkpoint::supra2::SD_VAE;

        // Neither component installed.
        let error = plan(&settings, &checkpoint, SUPRA2_SOURCE_FILE)
            .await
            .err()
            .expect("missing components are refused");
        assert_eq!(error.message, "Supra2-IMG components are not installed.");
        assert!(error.detail.contains(t5.repo), "{}", error.detail);

        // The text encoder at a DIFFERENT revision does not count.
        stage_snapshot(
            &data_dir,
            t5.repo,
            "0000000000000000000000000000000000000000",
            &["config.json"],
        );
        let error = plan(&settings, &checkpoint, SUPRA2_SOURCE_FILE)
            .await
            .err()
            .expect("an unpinned text encoder snapshot is refused");
        assert!(error.detail.contains(t5.revision), "{}", error.detail);

        let t5_dir = stage_snapshot(&data_dir, t5.repo, t5.revision, &["config.json"]);
        let error = plan(&settings, &checkpoint, SUPRA2_SOURCE_FILE)
            .await
            .err()
            .expect("a missing VAE is refused");
        assert!(error.detail.contains(vae.repo), "{}", error.detail);

        let vae_dir = stage_snapshot(&data_dir, vae.repo, vae.revision, &["config.json"]);
        match plan(&settings, &checkpoint, SUPRA2_SOURCE_FILE).await {
            Ok(ConvertPlan::Supra2 {
                source_file,
                text_encoder_dir,
                vae_dir: resolved_vae_dir,
            }) => {
                assert_eq!(source_file, checkpoint.join(SUPRA2_SOURCE_FILE));
                assert_eq!(text_encoder_dir, t5_dir);
                assert_eq!(resolved_vae_dir, vae_dir);
            }
            Ok(_) => panic!("supra2_img must resolve the Supra2 plan"),
            Err(error) => panic!("unexpected refusal: {} {}", error.message, error.detail),
        }
    }

    /// A component that fails its pinned size/SHA-256 check stops the convert before the bundle is
    /// written, with the inference verifier's reason surfaced, and leaves no install marker.
    #[test]
    fn supra2_convert_surfaces_component_verification_failure_and_writes_nothing() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let t5 = tmp.path().join("t5");
        let vae = tmp.path().join("vae");
        for dir in [&t5, &vae] {
            std::fs::create_dir_all(dir).expect("component dir");
            std::fs::write(dir.join("config.json"), b"{}").expect("config");
        }
        let out = tmp.path().join("out");
        let error = convert_supra2_bundle(&tmp.path().join("missing.pt"), &t5, &vae, &out)
            .expect_err("unverified components are refused");
        assert!(
            error.starts_with("Flan-T5-Base text encoder failed verification:"),
            "{error}"
        );
        assert!(
            !out.exists(),
            "nothing is written before the components verify"
        );
    }

    /// Real-weight install-time conversion from the local HF cache (CPU only: SHA-256 + tensor
    /// validation + safetensors write). `HF_HUB_CACHE=~/.cache/huggingface/hub cargo test -p
    /// sceneworks-worker supra2_real_weight_convert -- --ignored` with the three pinned snapshots in
    /// that hub (the resolver reads the env-selected hub, else the data dir's library).
    #[test]
    #[ignore = "needs HF_HUB_CACHE holding the pinned Supra2-IMG, Flan-T5-Base and SD VAE snapshots"]
    fn supra2_real_weight_convert_writes_a_loadable_bundle() {
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let snapshot = |pin: gen_core_checkpoint::supra2::ComponentPin| {
            huggingface_pinned_snapshot_dir(data_dir.path(), pin.repo, pin.revision)
                .unwrap_or_else(|| panic!("{}@{} is not cached", pin.repo, pin.revision))
        };
        let source = snapshot(gen_core_checkpoint::supra2::SUPRA2_SOURCE).join(SUPRA2_SOURCE_FILE);
        let t5 = snapshot(gen_core_checkpoint::supra2::FLAN_T5_BASE);
        let vae = snapshot(gen_core_checkpoint::supra2::SD_VAE);
        // `SUPRA2_BUNDLE_OUT` keeps the bundle for the acceptance matrix, which a LATER process
        // loads as `SUPRA2_BUNDLE_DIR` (the real-weight lanes convert once, in this separate step).
        let out = std::env::var_os("SUPRA2_BUNDLE_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.path().join("bundle"));
        convert_supra2_bundle(&source, &t5, &vae, &out).expect("real conversion succeeds");
        let bundle = gen_core_checkpoint::supra2::read_pinned_bundle(
            &out,
            gen_core_checkpoint::supra2::BundleCheck::Digests,
        )
        .expect("the bundle reads back under the pin contract");
        assert_eq!(bundle.manifest.provider_id, "supra2_img");
        assert!(
            out.join("model_index.json").is_file(),
            "install marker written"
        );
    }
}

#[cfg(test)]
mod anima_convert_tests {
    use super::*;

    /// The "30 GB trap" guard (sc-10517): an Anima install pulls EXACTLY the enumerated source files —
    /// three variant DiTs + the two shared components — never the bare repo (which holds SEVEN full
    /// bf16 DiTs, ~30 GB). Every path is a concrete file under `split_files/`, never an empty/repo-only
    /// entry (a whole-snapshot pull), and the union is the FIVE distinct files the manifest
    /// `downloads[].files` (sc-10523) must enumerate. A per-variant install is 3 of these (its DiT + the
    /// two shared), ~5.6 GB.
    #[test]
    fn source_file_plan_is_five_exact_paths_never_the_bare_repo() {
        let all: Vec<&str> = ANIMA_DIT_FILES
            .iter()
            .copied()
            .chain([ANIMA_TE_FILE, ANIMA_VAE_FILE])
            .collect();
        assert_eq!(all.len(), 5, "exactly 5 source files (3 DiT + TE + VAE)");
        for path in &all {
            // A concrete nested file — never the bare repo (which a whole-snapshot pull, i.e. an empty
            // `files` list, would be).
            assert!(!path.trim().is_empty(), "no empty (bare-repo) path");
            assert!(
                path.starts_with("split_files/") && path.ends_with(".safetensors"),
                "{path} must be a concrete split_files/*.safetensors path"
            );
        }
        assert_eq!(ANIMA_DIT_FILES.len(), 3, "the three shipped variant DiTs");
        for dit in ANIMA_DIT_FILES {
            assert!(dit.contains("/diffusion_models/anima-"), "{dit}");
            // The 3 preview cuts and aesthetic-v1.0b are explicitly out of scope.
            assert!(
                !dit.contains("preview") && !dit.contains("v1.0b"),
                "{dit} is an out-of-scope cut"
            );
        }
        let mut uniq = all.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            5,
            "the 5 source paths are distinct (no double-pull)"
        );
    }

    /// The "prefix trap" guard (sc-10517): the DiT quant-target predicate keeps the bundled
    /// `AnimaTextConditioner` DENSE for ALL THREE checkpoint roots — base roots the DiT at `net`,
    /// turbo/aesthetic at `model.diffusion_model`. The story's original "split on `net.llm_adapter.`"
    /// would have left the `model.diffusion_model.llm_adapter.*` conditioner UNPROTECTED (packed to Q4,
    /// dropping 134.7M params to garbage). This exercises BOTH roots, so a test covering only `net`
    /// would NOT have caught that hardcoded-prefix bug — this one does.
    #[test]
    fn quant_target_keeps_conditioner_dense_for_both_dit_roots() {
        for root in ["net", "model.diffusion_model"] {
            // Bundled conditioner projections — kept DENSE for both roots.
            assert!(!anima_is_dit_quant_target(&format!(
                "{root}.llm_adapter.blocks.0.self_attn.q_proj"
            )));
            assert!(!anima_is_dit_quant_target(&format!(
                "{root}.llm_adapter.out_proj"
            )));
            assert!(!anima_is_dit_quant_target(&format!(
                "{root}.llm_adapter.embed"
            )));
            // Cosmos DiT projections — PACKED for both roots.
            assert!(anima_is_dit_quant_target(&format!(
                "{root}.blocks.0.self_attn.q_proj"
            )));
            assert!(anima_is_dit_quant_target(&format!(
                "{root}.final_layer.linear"
            )));
            assert!(anima_is_dit_quant_target(&format!(
                "{root}.t_embedder.1.linear_1"
            )));
        }
    }

    /// Equivalence pin for the Anima DiT quant-target partition. The worker no longer duplicates the
    /// packer: `convert_anima_prequant` calls `runtime_macos::providers::anima::quantize_anima_dit` directly (sc-10580),
    /// so the crate's `is_dit_quant_target` is now the single source of the "DiT ⇒ quantize, bundled
    /// conditioner ⇒ dense" policy. This test keeps a hand-written table as an INDEPENDENT pin and, on
    /// macOS (where the crate is linked), asserts the real `runtime_macos::providers::anima::is_dit_quant_target` still
    /// matches it — so a future mlx-gen rev bump that silently repartitions the checkpoint (e.g. drops
    /// the `!contains("llm_adapter.")` rule and packs the conditioner to Q4) fails HERE. The
    /// `anima_is_dit_quant_target` local copy is the platform-independent half of the pin. Covers BOTH
    /// checkpoint roots (`net` = base, `model.diffusion_model` = aesthetic/turbo) and BOTH conditioner
    /// forms; see `mlx-gen-anima/src/convert.rs`.
    #[test]
    fn anima_dit_quant_predicate_matches_mlx_gen_anima_twin() {
        // (key minus `.weight`, expected) — true ⇒ pack as Cosmos DiT, false ⇒ keep dense.
        let cases: &[(&str, bool)] = &[
            // Bundled AnimaTextConditioner (`…llm_adapter.*`) — DENSE on BOTH roots / BOTH forms.
            ("net.llm_adapter.blocks.0.self_attn.q_proj", false),
            ("net.llm_adapter.out_proj", false),
            ("model.diffusion_model.llm_adapter.blocks.0.mlp.fc1", false),
            ("model.diffusion_model.llm_adapter.embed", false),
            // Cosmos DiT — PACKED on BOTH roots.
            ("net.blocks.0.self_attn.q_proj", true),
            ("net.final_layer.linear", true),
            ("model.diffusion_model.blocks.0.self_attn.q_proj", true),
            ("model.diffusion_model.t_embedder.1.linear_1", true),
        ];
        for (key, expected) in cases {
            assert_eq!(
                anima_is_dit_quant_target(key),
                *expected,
                "partition drift for `{key}` — must match runtime_macos::providers::anima::convert::is_dit_quant_target"
            );
            // On macOS the crate is linked: assert the REAL predicate the converter now delegates to
            // (`convert_anima_prequant` → `runtime_macos::providers::anima::quantize_anima_dit` → this) still matches the
            // pinned table, so a future mlx-gen rev can't silently repartition the checkpoint.
            #[cfg(target_os = "macos")]
            assert_eq!(
                runtime_macos::providers::anima::is_dit_quant_target(key),
                *expected,
                "crate drift for `{key}` — runtime_macos::providers::anima::is_dit_quant_target diverged from the pin"
            );
        }
    }
}

/// sc-10573 drift guard (the worker half of the convert-gap allow-list check). Pins the shared
/// `NATIVE_CONVERTERS` registry to the actual `resolve_convert_plan` arms so the two can never drift:
/// the convert-gap gate in `sceneworks-core` derives its allow-list from that same const, so keeping
/// it in lockstep with the arms here keeps the gate no stricter (and no looser) than what the worker
/// can really convert.
#[cfg(test)]
mod convert_registry_tests {
    use super::*;

    /// A `Settings` with just enough to construct an `ApiClient`; every download/limit field is inert
    /// for `resolve_convert_plan`'s converter-recognition path (no network is reached — each arm
    /// returns its own missing-source error before any fetch).
    fn minimal_settings(data_dir: PathBuf) -> Settings {
        Settings {
            api_url: "http://127.0.0.1:0".to_owned(),
            config_dir: PathBuf::from("config"),
            worker_id: "test-worker".to_owned(),
            is_child_worker: true,
            heartbeat_seconds: 5,
            huggingface_base_url: "http://127.0.0.1:0".to_owned(),
            // Values irrelevant to this test (no URL work happens); the byte-limit defaults.
            max_lora_url_bytes: 8u64 * 1024 * 1024 * 1024,
            max_model_url_bytes: 256u64 * 1024 * 1024 * 1024,
            allow_private_lora_urls: true,
            backend_mlx_enabled: true,
            ..crate::Settings::for_test(data_dir)
        }
    }

    /// A `model_convert` `JobSnapshot` carrying `converter`. It also carries `sourceFile`/`baseRepo`
    /// so the arms that `required_payload_string(..)` those before their existence checks reach their
    /// own (converter-specific) failure — never an infra `?` error out of `resolve_convert_plan`.
    fn convert_job(converter: &str) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job-convert-registry-test",
            "type": "model_convert",
            "status": "running",
            "projectId": null,
            "projectName": null,
            "payload": {
                "modelId": "test_model",
                "sourceRepo": "org/test-repo",
                "converter": converter,
                "sourceFile": "model.safetensors",
                "baseRepo": "org/base-repo"
            },
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0.0,
            "stage": "queued",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "elapsedSeconds": null,
            "attempts": 0,
            "sourceJobId": null,
            "duplicateOfJobId": null,
            "cancelRequested": false,
            "createdAt": "2026-07-10T00:00:00Z",
            "updatedAt": "2026-07-10T00:00:00Z",
            "startedAt": null,
            "completedAt": null,
            "canceledAt": null,
            "lastHeartbeatAt": null
        }))
        .expect("valid model_convert JobSnapshot")
    }

    /// Every id in `NATIVE_CONVERTERS` is a REAL `resolve_convert_plan` arm — it resolves to a plan or
    /// an arm-specific error, never the "Unknown MLX converter." fallback — and a bogus id IS rejected
    /// as unknown. Failure-capable: adding a nonexistent id to the const (no arm) makes it fall to the
    /// "Unknown MLX converter." path and this test goes RED; the bogus-id case proves the check isn't
    /// vacuously accepting every string.
    #[tokio::test]
    async fn native_converters_match_resolve_convert_plan_arms() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let settings = minimal_settings(tmp.path().to_path_buf());
        let api = ApiClient::new(&settings);
        let checkpoint_dir = tmp.path();

        for &converter in NATIVE_CONVERTERS {
            let job = convert_job(converter);
            let resolved = resolve_convert_plan(&api, &settings, &job, checkpoint_dir, None)
                .await
                .expect("resolver runs without an infra error");
            if let Err(err) = &resolved {
                assert_ne!(
                    err.message, "Unknown MLX converter.",
                    "'{converter}' is in NATIVE_CONVERTERS but resolve_convert_plan has no arm for it \
                     — the gate would allow a convert the worker rejects (sc-10573)"
                );
            }
        }

        // A converter NOT in the registry is rejected as unknown — proving the check above is
        // meaningful and that the fallback still fires.
        let bogus = convert_job("definitely_not_a_real_converter");
        let resolved = resolve_convert_plan(&api, &settings, &bogus, checkpoint_dir, None)
            .await
            .expect("resolver runs without an infra error");
        match resolved {
            Err(err) => assert_eq!(err.message, "Unknown MLX converter."),
            Ok(_) => panic!("a bogus converter must not resolve to a plan"),
        }
    }

    /// A raw queue row can bypass the typed API, so output confinement is the worker's final trust
    /// boundary. Both clients deliberately point at unreachable endpoints: if heartbeat/status work
    /// or LTX planning/provisioning runs first, this returns an HTTP/download error instead. The
    /// absent data root also proves the rejected job created no model/provisioning state.
    #[tokio::test]
    async fn invalid_convert_output_wins_before_api_or_ltx_provisioning_side_effects() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let data_dir = tmp.path().join("data");
        let settings = minimal_settings(data_dir.clone());
        let api = ApiClient::new(&settings);
        let mut job = convert_job("ltx_video");
        job.payload.insert(
            "outputDir".to_owned(),
            Value::String(tmp.path().join("outside").display().to_string()),
        );

        assert!(!data_dir.exists(), "fixture starts without app model state");
        let error = run_model_convert_job(&api, &settings, &job)
            .await
            .expect_err("invalid output must fail before any API or download access");

        assert!(
            matches!(error, WorkerError::InvalidPayload(ref detail) if detail.contains("direct child")),
            "output confinement must win error precedence, got {error}"
        );
        assert!(
            !data_dir.exists(),
            "invalid output must not create checkpoint, conversion, or provisioning state"
        );
    }
}

/// Offline quant-matrix tier builder (sc-8513, epic 8506) — the generalization of the FLUX.2-dev
/// pilot's one-off builder into a per-converter dispatch over the SAME production `ConvertPlan`
/// converters, so each hosted tier is byte-equivalent to the install-time output.
///
/// Produces ONE complete, HF-uploadable turnkey subdir (`q4/`, `q8/`, or `bf16/`) from a cached
/// dense diffusers source:
/// - `q4`/`q8`: run the production converter (packs `transformer/`, symlinks the dense TE/VAE/
///   tokenizer through), then dereference those symlinks into real files (HF upload does not follow
///   cache blob symlinks across a move).
/// - `bf16`: mirror the dense diffusers tree as-is (no quantize) — the dense backbone IS the bf16
///   tier.
///
/// `#[ignore]`: needs the cached dense source snapshot + Metal. Run ONE tier at a time (bounds the
/// transformer quantize disk peak), e.g.:
/// ```text
/// SC8513_CONVERTER=sd3_5_large_quant SC8513_BITS=4 \
///   SC8513_SRC="$HOME/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large/snapshots/<rev>" \
///   SC8513_OUT="$HOME/sc8513/sd3.5-large-mlx" \
///   cargo test -p sceneworks-worker --release tier_builder::build_tier -- --ignored --nocapture
/// ```
/// `SC8513_BITS<=0` builds the bf16 tier. New architectures: add a `SC8513_CONVERTER` arm mapping to
/// the model's production converter + its dense-reuse subdir list.
#[cfg(all(test, target_os = "macos"))]
mod tier_builder {
    use super::*;
    use std::path::{Path, PathBuf};

    fn env_path(key: &str) -> PathBuf {
        PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
    }

    /// Recursively copy `src` → `dst`, resolving any symlink (incl. HF-cache blob symlinks) to its
    /// real bytes so the output is a self-contained tree of real files.
    fn copy_tree_deref(src: &Path, dst: &Path) {
        if src.is_dir() {
            std::fs::create_dir_all(dst).unwrap();
            for entry in std::fs::read_dir(src).unwrap() {
                let entry = entry.unwrap();
                copy_tree_deref(&entry.path(), &dst.join(entry.file_name()));
            }
        } else {
            let real = std::fs::canonicalize(src).unwrap();
            std::fs::copy(&real, dst).unwrap();
        }
    }

    /// Replace every symlink under `dir` (the dense TE/VAE/tokenizer the converter borrowed from the
    /// source) with a real copy of its canonical target, in place.
    fn deref_symlinks(dir: &Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let ty = std::fs::symlink_metadata(&path).unwrap().file_type();
            if ty.is_symlink() {
                let real = std::fs::canonicalize(&path).unwrap();
                std::fs::remove_file(&path).unwrap();
                if real.is_dir() {
                    copy_tree_deref(&real, &path);
                } else {
                    std::fs::copy(&real, &path).unwrap();
                }
            } else if ty.is_dir() {
                deref_symlinks(&path);
            }
        }
    }

    /// The diffusers subdirs a converter reuses DENSE (also the extra bf16-mirror set beyond
    /// `transformer/`). `scheduler` is optional metadata and skipped when absent.
    fn dense_subs(converter: &str) -> &'static [&'static str] {
        match converter {
            "sd3_5_large_quant" | "sd3_5_large_turbo_quant" | "sd3_5_medium_quant" => &[
                "text_encoder",
                "text_encoder_2",
                "text_encoder_3",
                "tokenizer",
                "tokenizer_2",
                "tokenizer_3",
                "vae",
                "scheduler",
                "model_index.json",
            ],
            "flux2_dev_quant" | "flux2_klein_quant" => {
                &["text_encoder", "vae", "tokenizer", "model_index.json"]
            }
            other => panic!("unknown SC8513_CONVERTER {other}"),
        }
    }

    #[test]
    #[ignore = "real-weight: cached dense source snapshot + Metal"]
    fn build_tier() {
        let converter = std::env::var("SC8513_CONVERTER").expect("set SC8513_CONVERTER");
        let bits: i32 = std::env::var("SC8513_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let group_size: i32 = std::env::var("SC8513_GROUP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let src = env_path("SC8513_SRC");
        let out_root = env_path("SC8513_OUT");
        let tier = if bits <= 0 {
            "bf16".to_owned()
        } else {
            format!("q{bits}")
        };
        let out = out_root.join(&tier);
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        eprintln!(
            "[sc-8513] {converter} {tier} (bits {bits}, group {group_size}) {} -> {}",
            src.display(),
            out.display()
        );

        // Boogu (sc-8513, epic 8506): `assemble_quantized_snapshot` builds the whole turnkey in one
        // call — packs transformer/ + mllm/ and copies the dense mllm/tokenizer.json + vae/ as real
        // files — so there are no symlinks to dereference. Q4/Q8 only (bf16 ships dense separately).
        if converter == "boogu_assemble" {
            assert!(
                bits > 0,
                "boogu assemble builds a packed Q tier (set SC8513_BITS=4 or 8)"
            );
            runtime_macos::providers::boogu::convert::assemble_quantized_snapshot(&src, &out, bits)
                .unwrap_or_else(|error| panic!("boogu assemble failed: {error}"));
            eprintln!("[sc-8513] DONE {tier} at {}", out.display());
            return;
        }

        if bits <= 0 {
            // bf16: the dense backbone IS the tier — mirror transformer + the dense-reuse subdirs.
            copy_tree_deref(&src.join("transformer"), &out.join("transformer"));
            eprintln!("[sc-8513]   bf16 transformer mirrored");
            for sub in dense_subs(&converter) {
                let s = src.join(sub);
                if !s.exists() {
                    assert_eq!(*sub, "scheduler", "bf16 source missing required `{sub}`");
                    continue;
                }
                copy_tree_deref(&s, &out.join(sub));
                eprintln!("[sc-8513]   bf16 {sub} mirrored");
            }
        } else {
            // q4/q8: production converter packs transformer/ + symlinks dense parts; then realize.
            let result = match converter.as_str() {
                "sd3_5_large_quant" => {
                    convert_sd3_prequant(&src, &out, Sd3Variant::Large, bits, group_size)
                }
                "sd3_5_large_turbo_quant" => {
                    convert_sd3_prequant(&src, &out, Sd3Variant::LargeTurbo, bits, group_size)
                }
                "sd3_5_medium_quant" => {
                    convert_sd3_prequant(&src, &out, Sd3Variant::Medium, bits, group_size)
                }
                // FLUX.2-dev (Mistral TE) and FLUX.2-klein (Qwen3 TE) share the family
                // converter. `quantize_flux2_text_encoder_dir`'s `is_target` matches only Mistral
                // layer names, so on klein the Qwen3 TE stays DENSE bf16 — exactly epic 8506's
                // "quantize the transformer, keep the TE bf16" for a non-standard TE (sc-8711).
                // The same call yields a packed transformer + dense bf16 Qwen3 TE; the worker loads
                // it with Quant::None (`mlx.denseTextEncoderTier`) so the dense TE is never re-quantized.
                "flux2_dev_quant" | "flux2_klein_quant" => {
                    convert_flux2_dev_prequant(&src, &out, bits, group_size)
                }
                other => panic!("unknown SC8513_CONVERTER {other}"),
            };
            result.unwrap_or_else(|error| panic!("production convert failed: {error}"));
            eprintln!("[sc-8513]   transformer packed; dereferencing dense symlinks");
            deref_symlinks(&out);
        }
        eprintln!("[sc-8513] DONE {tier} at {}", out.display());
    }
}

/// Generic co-requisite → component-provisioning seam (epic 13657, sc-13679). Weights-free and
/// platform-neutral: it drives off a synthetic `ModelDescriptor` + manifest `Value` + an isolated
/// temp HF cache, so it runs on every lane (macOS `cargo test` AND the Linux parity clippy build).
#[cfg(test)]
mod co_requisite_tests {
    #[test]
    fn ltx_import_base_model_gate_rejects_only_unstamped_ltx() {
        let unstamped = JsonObject::new();
        assert!(ltx_import_base_model_error(Some("ltx-video"), &unstamped)
            .is_some_and(|detail| detail.contains("require an exact base model")));
        assert!(ltx_import_base_model_error(Some("wan-video"), &unstamped).is_none());

        let mut stamped = JsonObject::new();
        stamped.insert("baseModel".to_owned(), json!("ltx_2_5"));
        assert!(ltx_import_base_model_error(Some("ltx-video"), &stamped).is_none());
        stamped.insert("baseModel".to_owned(), json!("wan_2_2"));
        assert!(ltx_import_base_model_error(Some("ltx-video"), &stamped)
            .is_some_and(|detail| detail.contains("got wan_2_2")));
    }

    use super::*;

    /// A `Settings` whose only field `resolve_co_requisites` reads is `data_dir`; everything else is
    /// inert (no network, no engine).
    fn settings_at(data_dir: PathBuf) -> Settings {
        Settings {
            api_url: "http://127.0.0.1:0".to_owned(),
            config_dir: PathBuf::from("config"),
            worker_id: "sc-13679".to_owned(),
            is_child_worker: true,
            heartbeat_seconds: 5,
            huggingface_base_url: "http://127.0.0.1:0".to_owned(),
            max_lora_url_bytes: 0,
            max_model_url_bytes: 0,
            ..crate::Settings::for_test(data_dir)
        }
    }

    /// The chatterbox descriptor shape the resolver is driven by: it advertises the two named
    /// components (`perth`, `voice_embedding`) exactly as the real `chatterbox_tts` descriptor does at
    /// the inference pin (`required_components: &["perth", "voice_embedding"]`).
    fn chatterbox_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "chatterbox_tts",
            family: "chatterbox",
            backend: "candle",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &["perth", "voice_embedding"],
            control_kinds: None,
        }
    }

    /// The chatterbox manifest entry shape the resolver reads: the two `coRequisite` downloads carry
    /// an explicit `componentId` mapping repo → component id (never inferred from the repo name).
    fn chatterbox_manifest_entry() -> Value {
        json!({
            "id": "chatterbox_tts",
            "downloads": [
                { "provider": "huggingface", "repo": "ResembleAI/chatterbox",
                  "files": ["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"] },
                { "provider": "huggingface", "repo": "ResembleAI/chatterbox",
                  "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", "coRequisite": true,
                  "componentId": "voice_embedding", "files": ["ve.safetensors"] },
                { "provider": "huggingface", "repo": "SceneWorks/perth-implicit",
                  "revision": "80b60f9caead09b8d3b512bda0b24038f28c08ec", "coRequisite": true,
                  "componentId": "perth", "files": ["perth_implicit.safetensors"] }
            ]
        })
    }

    /// Materialize `<file>` inside `repo`'s cached HF snapshot under `data_dir`, so the cache-only
    /// `huggingface_snapshot_dir` resolves it — the disk state a completed co-requisite install leaves.
    fn stage_snapshot_file(data_dir: &Path, repo: &str, revision: &str, file: &str) -> PathBuf {
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).expect("create snapshot dir");
        let path = snapshot.join(file);
        // Some component files live under a subdir of the snapshot (mmaudio's `weights/…`,
        // `ext_weights/…` — sc-13684); create the parent so the write succeeds. A no-op for the
        // single-level component files (ve.safetensors, tokenizer.json) whose parent is the snapshot.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create component parent dir");
        }
        std::fs::write(&path, b"weights").expect("write staged component file");
        path
    }

    /// The optional, on-demand `sft_cover` component seam (sc-13821): acestep's Cover snapshot is a
    /// `coRequisite` deliberately kept OUT of `required_components`, so [`resolve_co_requisites`] never
    /// stages it. [`resolve_optional_component`] resolves it on demand — `Some(Dir)` when installed,
    /// `None` (never an error) when undeclared or absent, so an optional component is never
    /// install-gating (a `soft` co-requisite keeps text2music + the region edits usable).
    #[test]
    fn resolve_optional_component_resolves_sft_cover_only_when_declared_and_installed() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "ACE-Step/acestep-v15-xl-sft-diffusers";
        let revision = "4bf7b60a63b27144f539f980927eeb89f5f912b0";
        let manifest = json!({
            "id": "acestep_v15_turbo",
            "downloads": [
                { "provider": "huggingface", "repo": "ACE-Step/acestep-v15-xl-turbo-diffusers",
                  "revision": "200ba991ae448051e14b0183157e35c2d27c9fb0" },
                { "provider": "huggingface", "repo": repo, "revision": revision,
                  "coRequisite": true, "required": "soft", "componentId": "sft_cover",
                  "files": ["transformer/*", "audio_tokenizer/*", "audio_token_detokenizer/*"] }
            ]
        });
        let settings = settings_at(data_dir.path().to_path_buf());

        // Declared but NOT installed → None (the caller no-ops; never a hard error that would pre-empt
        // the engine's own on-demand Cover error and, on the pre-adoption pin, its self-fetch).
        assert!(
            resolve_optional_component(&manifest, "sft_cover", &settings).is_none(),
            "an absent optional component resolves to None, not an error"
        );
        // A component the manifest does not declare → None.
        assert!(
            resolve_optional_component(&manifest, "not_a_component", &settings).is_none(),
            "an undeclared component resolves to None"
        );

        // Install it → resolves to the PINNED snapshot dir (a multi-file/glob component ⇒ Dir root).
        stage_snapshot_file(
            data_dir.path(),
            repo,
            revision,
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        );
        match resolve_optional_component(&manifest, "sft_cover", &settings) {
            Some(gen_core::WeightsSource::Dir(dir)) => assert!(
                dir.ends_with(revision),
                "an installed sft_cover resolves to the pinned snapshots/<sha>/ dir, got {dir:?}"
            ),
            other => panic!("an installed sft_cover resolves to a Dir, got {other:?}"),
        }
    }

    #[test]
    fn optional_predictor_load_rejects_same_size_wrong_digest_and_ignores_main() {
        use sha2::{Digest, Sha256};
        let _env = crate::test_env::isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let settings = settings_at(data.path().into());
        let repo = "test/predictor";
        let revision = "a".repeat(40);
        let file = "predictor.safetensors";
        let row = json!({"repo": repo, "revision": revision, "files": [file],
            "coRequisite": true, "required": "soft", "componentId": "veda_predictor",
            "expectedSizeBytes": 4, "expectedSha256": format!("{:x}", Sha256::digest(b"good"))});
        let manifest = json!({"downloads":[row]});
        assert!(resolve_optional_component(&manifest, "veda_predictor", &settings).is_none());
        let path = stage_snapshot_file(data.path(), repo, &revision, file);
        std::fs::write(&path, b"good").unwrap();
        assert!(
            matches!(resolve_optional_component(&manifest,"veda_predictor",&settings),Some(gen_core::WeightsSource::File(p)) if p == path)
        );
        std::fs::write(&path, b"evil").unwrap();
        assert!(resolve_optional_component(&manifest, "veda_predictor", &settings).is_none());
        std::fs::remove_file(&path).unwrap();
        let main = stage_snapshot_file(data.path(), repo, &"b".repeat(40), file);
        std::fs::write(main, b"good").unwrap();
        assert!(resolve_optional_component(&manifest, "veda_predictor", &settings).is_none());
    }

    /// Soft co-requisites are install companions for request-specific features. Even when their
    /// bytes are already cached, neither backend's generic descriptor resolver may reinterpret one
    /// as an unconditional component and attach it to every ordinary job.
    #[test]
    fn resolve_co_requisites_rejects_soft_components_on_both_backends() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "xinsir/controlnet-openpose-sdxl-1.0";
        let revision = "23f966cd5cfdd3f7729c903e243d87152162d2b7";
        stage_snapshot_file(
            data_dir.path(),
            repo,
            revision,
            "diffusion_pytorch_model.safetensors",
        );
        let manifest = json!({
            "id": "sdxl_probe",
            "downloads": [{
                "provider": "huggingface",
                "repo": repo,
                "revision": revision,
                "coRequisite": true,
                "required": "soft",
                "componentId": "controlnet_openpose",
                "files": ["diffusion_pytorch_model.safetensors"]
            }]
        });
        let settings = settings_at(data_dir.path().to_path_buf());

        for backend in ["mlx", "candle"] {
            let descriptor = gen_core::ModelDescriptor {
                id: "sdxl_probe",
                family: "sdxl",
                backend,
                modality: gen_core::Modality::Image,
                capabilities: gen_core::Capabilities::default(),
                encoder_contract: None,
                denoiser_output_latent_space: None,
                required_components: &["controlnet_openpose"],
                control_kinds: None,
            };
            let error = resolve_co_requisites(&descriptor, &manifest, &settings)
                .expect_err("a soft component cannot satisfy required_components");
            assert!(
                matches!(error, WorkerError::InvalidPayload(ref message)
                    if message.contains("required: soft")
                        && message.contains("request-specific route")),
                "{backend}: an installed soft component must still be rejected by the generic seam, got {error:?}"
            );
        }
    }

    /// sc-13583 / F-002 (co-requisite seam, 5th site): `revision` reaches
    /// [`huggingface_pinned_snapshot_dir`] straight from the payload `modelManifestEntry`. A `..`
    /// revision must NOT resolve to a directory OUTSIDE `snapshots/`, even when that directory exists
    /// on disk (it would pass `is_dir()` and, on Windows, get its symlinks rewritten to hardlinks).
    #[test]
    fn huggingface_pinned_snapshot_dir_confines_a_traversal_revision() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "ResembleAI/chatterbox";

        // A real pinned snapshot still resolves — proving the guard is a confinement, not a blanket
        // reject of every non-`main` revision.
        let revision = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";
        stage_snapshot_file(data_dir.path(), repo, revision, "ve.safetensors");
        assert!(
            huggingface_pinned_snapshot_dir(data_dir.path(), repo, revision).is_some(),
            "a real pinned snapshot must still resolve"
        );

        // Plant a real directory a level ABOVE snapshots/ that a `../` revision would land on …
        let escaped = huggingface_repo_cache_path(data_dir.path(), repo)
            .expect("repo cache path resolves")
            .join("escaped");
        std::fs::create_dir_all(&escaped).expect("escaped dir");
        // … and confirm the traversal revision does not resolve to it.
        assert_eq!(
            huggingface_pinned_snapshot_dir(data_dir.path(), repo, "../escaped"),
            None,
            "a `..` revision must not resolve to a dir outside snapshots/"
        );
        for invalid in [
            "abc123",
            "5BB1F6EE58E50C3B8D408BC82A6D3740C2DB6E18",
            "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e1g",
        ] {
            assert_eq!(
                huggingface_pinned_snapshot_dir(data_dir.path(), repo, invalid),
                None,
                "a pinned revision must be exactly 40 lowercase hex characters: {invalid}"
            );
        }
    }

    #[test]
    fn resolve_co_requisites_rejects_a_torn_multi_file_component() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "owner/multi-component";
        let revision = "0123456789abcdef0123456789abcdef01234567";
        stage_snapshot_file(data_dir.path(), repo, revision, "weights/a.bin");
        let descriptor = gen_core::ModelDescriptor {
            id: "multi_model",
            family: "multi",
            backend: "candle",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &["bundle"],
            control_kinds: None,
        };
        let manifest = json!({
            "id": "multi_model",
            "downloads": [{
                "provider": "huggingface",
                "repo": repo,
                "revision": revision,
                "coRequisite": true,
                "componentId": "bundle",
                "files": ["weights/a.bin", "weights/b.bin"]
            }]
        });
        let settings = settings_at(data_dir.path().to_path_buf());

        let error = resolve_co_requisites(&descriptor, &manifest, &settings)
            .expect_err("a missing member makes the component incomplete");
        assert!(
            matches!(error, WorkerError::InvalidPayload(ref message) if message.contains("weights/b.bin")),
            "the repair error must identify the missing member, got {error:?}"
        );

        stage_snapshot_file(data_dir.path(), repo, revision, "weights/b.bin");
        assert!(
            matches!(
                resolve_co_requisites(&descriptor, &manifest, &settings)
                    .expect("the complete component resolves")
                    .get("bundle"),
                Some(gen_core::WeightsSource::Dir(path)) if path.ends_with(revision)
            ),
            "a complete multi-file component stages its snapshot root"
        );
    }

    /// sc-13583 (co-requisite seam): a single-entry `files: ["<../.. path>"]` in the payload
    /// `modelManifestEntry` must not be joined onto the snapshot to stage an arbitrary host file as a
    /// required component's weights — `resolve_co_requisites` must reject it via `safe_join`.
    #[test]
    fn resolve_co_requisites_confines_a_traversal_component_file() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "ResembleAI/chatterbox";
        let revision = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";
        // Snapshot present, plus a target file one level ABOVE the snapshot that a `../` entry hits.
        stage_snapshot_file(data_dir.path(), repo, revision, "placeholder.bin");
        let outside = huggingface_repo_cache_path(data_dir.path(), repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join("outside.bin");
        std::fs::write(&outside, b"secret").expect("plant outside file");

        let descriptor = gen_core::ModelDescriptor {
            id: "trav_probe",
            family: "chatterbox",
            backend: "candle",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &["voice_embedding"],
            control_kinds: None,
        };
        let manifest = json!({
            "id": "trav_probe",
            "downloads": [
                { "provider": "huggingface", "repo": repo, "revision": revision, "coRequisite": true,
                  "componentId": "voice_embedding", "files": ["../outside.bin"] }
            ]
        });
        let settings = settings_at(data_dir.path().to_path_buf());
        // The outside file EXISTS, so without the guard the join would resolve it to a real File
        // (Ok), not the missing-file error — `expect_err` therefore discriminates the guard itself.
        let error = resolve_co_requisites(&descriptor, &manifest, &settings)
            .expect_err("a `..` component file entry is rejected");
        assert!(
            matches!(&error, WorkerError::InvalidPayload(message) if message.contains("Unsafe snapshot path")),
            "expected a safe_join rejection, got {error:?}"
        );
    }

    /// sc-13583 (co-requisite seam): the optional-component twin of the guard above — a `..` file
    /// entry resolves to `None` (component simply not staged), never an out-of-snapshot host file.
    #[test]
    fn resolve_optional_component_confines_a_traversal_component_file() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "ACE-Step/acestep-v15-xl-sft-diffusers";
        let revision = "4bf7b60a63b27144f539f980927eeb89f5f912b0";
        stage_snapshot_file(data_dir.path(), repo, revision, "placeholder.bin");
        let outside = huggingface_repo_cache_path(data_dir.path(), repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join("outside.bin");
        std::fs::write(&outside, b"secret").expect("plant outside file");

        let manifest = json!({
            "id": "acestep_v15_turbo",
            "downloads": [
                { "provider": "huggingface", "repo": repo, "revision": revision,
                  "coRequisite": true, "required": "soft", "componentId": "sft_cover",
                  "files": ["../outside.bin"] }
            ]
        });
        let settings = settings_at(data_dir.path().to_path_buf());
        assert!(
            resolve_optional_component(&manifest, "sft_cover", &settings).is_none(),
            "a `..` file entry must not stage a host file outside the snapshot"
        );
    }

    #[test]
    fn snapshot_resolution_skips_an_empty_refs_main_and_falls_back_to_the_populated_snapshot() {
        // A `refs/main` clobbered to an empty placeholder snapshot (the sc-13834 cache-pollution
        // shape) must NOT strand the model on that empty dir; the resolver falls back to the
        // most-populated cached sibling instead. Regression guard for the old "first dir wins" scan,
        // which happily returned an empty snapshot the moment `refs/main` pointed at one.
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "SceneWorks/krea-2-raw-mlx";
        let repo_dir =
            huggingface_repo_cache_path(data_dir.path(), repo).expect("repo cache path resolves");
        let snapshots = repo_dir.join("snapshots");

        // The real, fully materialized snapshot …
        let populated_revision = "b88090c7b88090c7b88090c7b88090c7b88090c7";
        let empty_revision = "abc12300abc12300abc12300abc12300abc12300";
        let populated = snapshots.join(populated_revision);
        std::fs::create_dir_all(populated.join("q8/transformer")).expect("populated tree");
        std::fs::write(
            populated.join("q8/transformer/diffusion_pytorch_model.safetensors"),
            b"real",
        )
        .expect("write real weight");
        // … alongside an EMPTY placeholder snapshot (tier dirs only, no files) …
        std::fs::create_dir_all(snapshots.join(empty_revision).join("q8/transformer"))
            .expect("empty placeholder");
        // … with `refs/main` clobbered to point at the empty one.
        std::fs::create_dir_all(repo_dir.join("refs")).expect("create refs");
        std::fs::write(repo_dir.join("refs").join("main"), empty_revision)
            .expect("write refs/main");

        let resolved = resolve_huggingface_snapshot_dir(data_dir.path(), repo)
            .expect("a populated snapshot exists, so resolution must succeed");
        assert_eq!(
            resolved, populated,
            "a refs/main pointing at an empty snapshot must fall back to the populated sibling"
        );

        // Sanity: a VALID `refs/main` (populated) is still honored on the fast path — the fallback
        // only engages when `refs/main` names an empty/absent snapshot.
        std::fs::write(repo_dir.join("refs").join("main"), populated_revision)
            .expect("repoint refs/main");
        assert_eq!(
            resolve_huggingface_snapshot_dir(data_dir.path(), repo).expect("still resolves"),
            populated,
            "a populated refs/main must resolve to exactly that snapshot"
        );
    }

    /// A snapshot directory whose name is not a 40-hex commit is still an installed snapshot.
    ///
    /// Routing discovery through the typed artifact seam (sc-19704) started admitting the on-disk
    /// directory name as if it were a caller-supplied revision, so `ArtifactSourceLibrary`'s
    /// immutability rule applied to it and every such snapshot resolved to `None` — the model read
    /// as not installed, silently, on every lane. This is reachable in production:
    /// `downloads::download_snapshot_into_cache` falls back to
    /// `commit.unwrap_or_else(|| revision.to_owned())`, so an endpoint that omits `X-Repo-Commit`
    /// materializes a complete install under `snapshots/main`.
    ///
    /// Both arms are asserted because the `refs/main` arm *returns*: when it rejected the name it
    /// did not fall through to the scan, so a repository with a valid `refs/main` resolved to
    /// nothing even though the populated snapshot was sitting right there.
    #[test]
    fn snapshot_resolution_accepts_a_snapshot_named_by_a_mutable_revision() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "SceneWorks/krea-2-raw-mlx";
        let repo_dir =
            huggingface_repo_cache_path(data_dir.path(), repo).expect("repo cache path resolves");
        let snapshot = repo_dir.join("snapshots").join("main");
        std::fs::create_dir_all(snapshot.join("q8/transformer")).expect("snapshot tree");
        std::fs::write(
            snapshot.join("q8/transformer/diffusion_pytorch_model.safetensors"),
            b"real",
        )
        .expect("write weight");

        // The scan arm: no `refs/main` at all, exactly like the seeded turnkey fixtures the candle
        // image lanes resolve against.
        assert_eq!(
            resolve_huggingface_snapshot_dir(data_dir.path(), repo)
                .expect("a populated snapshot exists, so resolution must succeed"),
            snapshot,
            "a snapshot directory named by a mutable revision is still installed"
        );

        // The `refs/main` fast-path arm, which is what a real mirror-endpoint install looks like:
        // the pointer and the snapshot directory both carry the mutable name.
        std::fs::create_dir_all(repo_dir.join("refs")).expect("create refs");
        std::fs::write(repo_dir.join("refs").join("main"), "main").expect("write refs/main");
        assert_eq!(
            resolve_huggingface_snapshot_dir(data_dir.path(), repo)
                .expect("refs/main names a populated snapshot, so resolution must succeed"),
            snapshot,
            "a refs/main naming a populated mutable snapshot must resolve to exactly that snapshot"
        );
    }

    /// A repository installed ONLY at a pinned `snapshots/<40-hex>` — no `refs/main` — still
    /// resolves through the UNPINNED repo-string seam every native LLM consumer uses.
    ///
    /// This is the production shape of a manifest download that carries a `revision` (F-029):
    /// `download_snapshot_into_cache` writes the receipt pointer as `refs/<revision>`, so a pinned
    /// install leaves `refs/main` ABSENT. The consumers, however, resolve by repo string with no
    /// revision at all — `prompt_refine_jobs` (`DEFAULT_REFINE_MODEL`), `caption_jobs`,
    /// `dataset_analysis_jobs` — so the pin's correctness depends entirely on the scan arm below
    /// finding the 40-hex snapshot after the `refs/main` read fails.
    ///
    /// The neighbouring mutable-name test exercises the OTHER branch of
    /// `discover_installed_snapshot_path` (a non-immutable directory name is admitted as an
    /// installed path); an immutable name is delegated to `discover_snapshot` instead, so a pinned
    /// install would go unguarded without this case. Asserted through
    /// `resolve_app_managed_model_dir`, the seam the jobs actually call, not just the inner
    /// resolver — a pinned model that resolves inwardly but is rejected at the job seam is still a
    /// model that cannot load.
    #[test]
    fn snapshot_resolution_finds_a_pinned_install_that_has_no_refs_main() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let repo = "TheDrummer/Anubis-Mini-8B-v1";
        let revision = "696f5b956f0511168d98cd32106299cebc3cc12b";
        let repo_dir =
            huggingface_repo_cache_path(data_dir.path(), repo).expect("repo cache path resolves");
        let snapshot = repo_dir.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");
        std::fs::write(
            snapshot.join("model-00001-of-00004.safetensors"),
            b"weights",
        )
        .expect("write weight");
        // The receipt pointer a PINNED download leaves — named by the revision, never `main`.
        std::fs::create_dir_all(repo_dir.join("refs")).expect("create refs");
        std::fs::write(repo_dir.join("refs").join(revision), revision).expect("write refs/<rev>");
        assert!(
            !repo_dir.join("refs").join("main").exists(),
            "the fixture must reproduce a pinned install: no refs/main"
        );

        assert_eq!(
            resolve_huggingface_snapshot_dir(data_dir.path(), repo)
                .expect("a pinned install is installed, so resolution must succeed"),
            snapshot,
        );
        assert_eq!(
            crate::paths::resolve_app_managed_model_dir(
                &settings_at(data_dir.path().to_path_buf()),
                repo,
                "prompt-refine model path",
            )
            .expect("the job seam must resolve a pinned install"),
            snapshot,
        );
    }

    #[test]
    fn present_co_requisites_resolve_to_a_component_map_matching_the_descriptor() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let ve = stage_snapshot_file(
            data_dir.path(),
            "ResembleAI/chatterbox",
            "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
            "ve.safetensors",
        );
        let perth = stage_snapshot_file(
            data_dir.path(),
            "SceneWorks/perth-implicit",
            "80b60f9caead09b8d3b512bda0b24038f28c08ec",
            "perth_implicit.safetensors",
        );

        let descriptor = chatterbox_descriptor();
        let components = resolve_co_requisites(
            &descriptor,
            &chatterbox_manifest_entry(),
            &settings_at(data_dir.path().to_path_buf()),
        )
        .expect("both co-requisites are installed, so resolution succeeds");

        // The resolved map's keys are EXACTLY the descriptor's advertised component ids — compared as
        // a SET, since callers key by id (order-independent) and a future descriptor whose ids are not
        // alphabetical must not false-fail against the BTreeMap's sorted key order. Still fully
        // discriminating: a missing or extra component id breaks set equality.
        let keys: std::collections::BTreeSet<&str> =
            components.keys().map(String::as_str).collect();
        let required: std::collections::BTreeSet<&str> =
            descriptor.required_components.iter().copied().collect();
        assert_eq!(
            keys, required,
            "the component map keys must match the descriptor's required_components (as a set)"
        );
        // Each id resolves to the single-file WeightsSource at the staged snapshot path.
        // (`WeightsSource` has no `PartialEq`, so match the `File` variant and compare its path.)
        let file_path = |source: Option<&gen_core::WeightsSource>| match source {
            Some(gen_core::WeightsSource::File(path)) => Some(path.clone()),
            _ => None,
        };
        assert_eq!(
            file_path(components.get("voice_embedding")),
            Some(ve),
            "voice_embedding maps to the staged ve.safetensors"
        );
        assert_eq!(
            file_path(components.get("perth")),
            Some(perth),
            "perth maps to the staged perth_implicit.safetensors"
        );
    }

    #[test]
    fn a_missing_co_requisite_fails_with_an_actionable_error_before_the_engine() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        // Stage ONLY voice_embedding; perth's snapshot is absent.
        stage_snapshot_file(
            data_dir.path(),
            "ResembleAI/chatterbox",
            "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
            "ve.safetensors",
        );

        let error = resolve_co_requisites(
            &chatterbox_descriptor(),
            &chatterbox_manifest_entry(),
            &settings_at(data_dir.path().to_path_buf()),
        )
        .expect_err("a missing co-requisite must fail the job before the engine load");

        let WorkerError::InvalidPayload(message) = error else {
            panic!(
                "a missing co-requisite must surface as user-facing InvalidPayload, got {error:?}"
            );
        };
        // Actionable: names the missing component id AND the repo to install/predownload.
        assert!(
            message.contains("perth") && message.contains("SceneWorks/perth-implicit"),
            "the error must name the missing component id + repo, got: {message}"
        );
    }

    #[test]
    fn a_required_component_with_no_matching_component_id_is_a_manifest_error() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        // A manifest entry whose co-requisites carry NO componentId → the perth id can't be mapped.
        let manifest_entry = json!({
            "id": "chatterbox_tts",
            "downloads": [
                { "provider": "huggingface", "repo": "SceneWorks/perth-implicit",
                  "revision": "80b60f9caead09b8d3b512bda0b24038f28c08ec", "coRequisite": true,
                  "files": ["perth_implicit.safetensors"] }
            ]
        });
        let error = resolve_co_requisites(
            &chatterbox_descriptor(),
            &manifest_entry,
            &settings_at(data_dir.path().to_path_buf()),
        )
        .expect_err("a required id with no componentId mapping is a manifest misconfiguration");
        let WorkerError::InvalidPayload(message) = error else {
            panic!("expected InvalidPayload, got {error:?}");
        };
        assert!(
            message.contains("componentId") && message.contains("perth"),
            "the error must name the unmapped component id + the missing componentId tag, got: {message}"
        );
    }

    #[test]
    fn an_empty_required_components_set_is_a_no_op() {
        let _env = crate::test_env::isolate_hf_cache();
        // Every current image/video model + single-file audio model advertises no components: the
        // resolver must return an empty map WITHOUT touching the filesystem (a nonexistent data_dir).
        let descriptor = gen_core::ModelDescriptor {
            id: "kokoro_82m",
            family: "kokoro",
            backend: "candle",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &[],
            control_kinds: None,
        };
        let components = resolve_co_requisites(
            &descriptor,
            &json!({ "id": "kokoro_82m", "downloads": [] }),
            &settings_at(PathBuf::from("/nonexistent/sc-13679")),
        )
        .expect("an empty required_components set never fails");
        assert!(
            components.is_empty(),
            "no components are staged for a model that requires none"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_sdxl_descriptor_stages_none_of_the_carried_candle_components() {
        let descriptor = crate::inference_runtime::media_descriptor("sdxl")
            .expect("the macOS MLX SDXL provider is registered");
        assert_eq!(descriptor.backend, "mlx");
        assert!(
            descriptor.required_components.is_empty(),
            "the self-contained MLX SDXL provider requires no caller-staged components"
        );

        let components = resolve_co_requisites(
            &descriptor,
            &builtin_manifest_entry("realvisxl"),
            &settings_at(PathBuf::from("/nonexistent/sc-20747-mlx-sdxl")),
        )
        .expect("an empty MLX descriptor ignores carried Candle-only component metadata");
        assert!(
            components.is_empty(),
            "the generic resolver stages no components for self-contained MLX SDXL"
        );
    }

    // --- SDXL shared components (epic 13657, sc-13682) -----------------------------------------------

    /// The candle `sdxl` generator descriptor shape at the inference pin: it advertises the three
    /// caller-staged SDXL components the edit / IP-Adapter / InstantID / trainer providers also consume.
    fn sdxl_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "sdxl",
            family: "sdxl",
            backend: "candle",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"],
            control_kinds: None,
        }
    }

    /// The LIVE `builtin.models.jsonc` entry for `model_id` — binds these tests to the SHIPPED manifest so
    /// a dropped/renamed SDXL co-requisite (or a `componentId` that stops matching the descriptor's
    /// `required_components`) fails HERE, not silently at candle render time.
    fn builtin_manifest_entry(model_id: &str) -> Value {
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc present");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin.models.jsonc parses");
        manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
            .cloned()
            .unwrap_or_else(|| panic!("builtin entry {model_id} present"))
    }

    /// `(repo, pinned revision, single file)` for each of the three SDXL components — the exact
    /// snapshots the manifest co-requisites pin (sc-13682, sc-13658 registry).
    const SDXL_COMPONENT_SNAPSHOTS: &[(&str, &str, &str)] = &[
        (
            "openai/clip-vit-large-patch14",
            "32bd64288804d66eefd0ccbe215aa642df71cc41",
            "tokenizer.json",
        ),
        (
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "743c27bd53dfe508a0ade0f50698f99b39d03bec",
            "tokenizer.json",
        ),
        (
            "madebyollin/sdxl-vae-fp16-fix",
            "207b116dae70ace3637169f1ddd2434b91b3a8cd",
            "diffusion_pytorch_model.safetensors",
        ),
    ];

    #[test]
    fn sdxl_co_requisites_resolve_all_three_from_every_live_candle_sdxl_entry() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        for (repo, revision, file) in SDXL_COMPONENT_SNAPSHOTS {
            stage_snapshot_file(data_dir.path(), repo, revision, file);
        }

        // Every Candle-SDXL base + InstantID declares the SAME three hard coRequisites by
        // `componentId`. The five pose backbones also declare soft OpenPose, which this generic
        // descriptor seam must ignore. Resolving against each live entry maps exactly the hard three.
        for model_id in [
            "sdxl",
            "realvisxl",
            "realvisxl_lightning",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
            "instantid_realvisxl",
        ] {
            let components = resolve_co_requisites(
                &sdxl_descriptor(),
                &builtin_manifest_entry(model_id),
                &settings_at(data_dir.path().to_path_buf()),
            )
            .unwrap_or_else(|error| {
                panic!("{model_id}: all three components are staged, so resolution must succeed: {error:?}")
            });

            // The map keys are EXACTLY the descriptor's advertised ids (order-independent — a BTreeMap).
            let mut keys: Vec<&str> = components.keys().map(String::as_str).collect();
            keys.sort_unstable();
            let mut want = sdxl_descriptor().required_components.to_vec();
            want.sort_unstable();
            assert_eq!(
                keys, want,
                "{model_id}: the component map keys must match the descriptor's required_components"
            );
            // Each single-file component resolves to a `File` at the staged snapshot path.
            assert!(
                matches!(
                    components.get("vae_fp16_fix"),
                    Some(gen_core::WeightsSource::File(_))
                ),
                "{model_id}: vae_fp16_fix resolves to its single-file snapshot"
            );
        }
    }

    #[test]
    fn a_missing_sdxl_component_fails_with_an_actionable_error_naming_id_and_repo() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        // Stage only the two tokenizers; the fp16-fix VAE snapshot is ABSENT.
        for (repo, revision, file) in &SDXL_COMPONENT_SNAPSHOTS[..2] {
            stage_snapshot_file(data_dir.path(), repo, revision, file);
        }

        let error = resolve_co_requisites(
            &sdxl_descriptor(),
            &builtin_manifest_entry("sdxl"),
            &settings_at(data_dir.path().to_path_buf()),
        )
        .expect_err("a missing SDXL component must fail the job before the engine load");

        let WorkerError::InvalidPayload(message) = error else {
            panic!("a missing component must surface as user-facing InvalidPayload, got {error:?}");
        };
        assert!(
            message.contains("vae_fp16_fix") && message.contains("madebyollin/sdxl-vae-fp16-fix"),
            "the error must name the missing component id + repo BEFORE engine load, got: {message}"
        );
    }

    // --- MOSS TTS codec component (epic 13678, sc-13681) ---------------------------------------------

    /// A candle MOSS TTS descriptor shape at the inference pin: it advertises the single caller-staged
    /// `codec` component (`candle-audio-moss-tts` / `candle-audio-moss-tts-realtime` both declare
    /// `required_components: &["codec"]`, `CODEC_COMPONENT_ID = "codec"`), which the worker resolves
    /// from the model's codec coRequisite instead of the provider self-fetching it (sc-13662).
    fn moss_descriptor(id: &'static str, family: &'static str) -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id,
            family,
            backend: "candle",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &["codec"],
            control_kinds: None,
        }
    }

    /// `(model_id, family, codec repo, pinned revision, a file in the codec snapshot)` — the exact codec
    /// coRequisite each SHIPPED MOSS TTS entry pins (sc-13681). MOSS-TTSD's XY_Tokenizer is a single-file
    /// checkpoint (resolves to a `File`); MOSS-TTS-Realtime's MOSS-Audio-Tokenizer is a multi-file
    /// snapshot directory (resolves to a `Dir`). Binds these tests to the LIVE `builtin.models.jsonc`, so
    /// a dropped codec coRequisite or a `componentId` that stops matching `required_components: ["codec"]`
    /// fails HERE, not silently at candle synth time.
    const MOSS_CODEC_SNAPSHOTS: &[(&str, &str, &str, &str, &str)] = &[
        (
            "moss_ttsd_v05",
            "moss_ttsd",
            "OpenMOSS-Team/XY_Tokenizer_TTSD_V0",
            "c83433728e698ed0698e88cb5096bc221fb8f8c5",
            "xy_tokenizer.ckpt",
        ),
        (
            "moss_tts_realtime",
            "moss_tts_realtime",
            "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "3cd226ba2947efa357ef453bcad111b6eafba782",
            "config.json",
        ),
    ];

    fn mage_descriptor(id: &'static str) -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id,
            family: "mage-flow",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &["text_encoder", "vae"],
            control_kinds: None,
        }
    }

    /// The six SHIPPED Mage rows, whose shared text encoder + VAE are per-tier co-requisites.
    const MAGE_MODEL_IDS: &[&str] = &[
        "mage_flow_base",
        "mage_flow",
        "mage_flow_turbo",
        "mage_flow_edit_base",
        "mage_flow_edit",
        "mage_flow_edit_turbo",
    ];

    /// Stage the shared components mirror's `<tier>/text_encoder/` + `<tier>/vae/` for one tier.
    fn stage_mage_components(data_dir: &Path, manifest: &Value, tier: &str) {
        for download in manifest["downloads"].as_array().expect("downloads") {
            if download.get("coRequisite").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            if download.get("variant").and_then(Value::as_str) != Some(tier) {
                continue;
            }
            let repo = download["repo"].as_str().expect("repo");
            let revision = download["revision"].as_str().expect("revision");
            let subdir = download["subdir"].as_str().expect("subdir");
            // One real file inside the component dir, so both the `files` glob check and the
            // `subdir` is_dir() check see a populated tree.
            stage_snapshot_file(
                data_dir,
                repo,
                revision,
                &format!("{subdir}/model.safetensors"),
            );
        }
    }

    /// sc-14980/sc-14979 — a per-tier co-requisite resolves to the tier's own component DIR.
    ///
    /// Discriminating in the way that matters: only the q4 tier is staged, and the assertion is that
    /// each component path ends in `q4/<component>`. A resolver that ignored `subdir` would hand
    /// back the snapshot ROOT (the engine would then look for `text_encoder/` at the top level and
    /// fail), and one that ignored `variant` would pick whichever row came first.
    #[test]
    fn mage_per_tier_co_requisites_resolve_to_the_matching_tier_component_dir() {
        for &model_id in MAGE_MODEL_IDS {
            let _env = crate::test_env::isolate_hf_cache();
            let data_dir = tempfile::tempdir().expect("temp data dir");
            let manifest = builtin_manifest_entry(model_id);
            stage_mage_components(data_dir.path(), &manifest, "q4");

            let components = resolve_co_requisites_for_tier(
                &mage_descriptor(model_id),
                &manifest,
                &settings_at(data_dir.path().to_path_buf()),
                Some("q4"),
            )
            .unwrap_or_else(|error| panic!("{model_id}: q4 components are staged: {error:?}"));

            let keys: Vec<&str> = components.keys().map(String::as_str).collect();
            assert_eq!(keys, vec!["text_encoder", "vae"], "{model_id}");
            for component in ["text_encoder", "vae"] {
                let gen_core::WeightsSource::Dir(dir) = &components[component] else {
                    panic!("{model_id}: {component} must stage as a directory");
                };
                assert!(
                    dir.ends_with(format!("q4/{component}")),
                    "{model_id}: {component} must resolve to the q4 component dir, got {}",
                    dir.display()
                );
                assert!(dir.is_dir(), "{model_id}: {component} dir must exist");
            }
        }
    }

    /// Asking for a tier whose shared components are NOT installed fails BEFORE the engine load,
    /// rather than silently handing back another tier's text encoder.
    #[test]
    fn mage_co_requisites_refuse_a_tier_that_is_not_installed() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let manifest = builtin_manifest_entry("mage_flow_base");
        stage_mage_components(data_dir.path(), &manifest, "q4");

        let error = resolve_co_requisites_for_tier(
            &mage_descriptor("mage_flow_base"),
            &manifest,
            &settings_at(data_dir.path().to_path_buf()),
            Some("bf16"),
        )
        .expect_err("bf16 components are not staged, so resolution must fail");
        let message = format!("{error:?}");
        assert!(
            message.contains("not installed") || message.contains("missing"),
            "the failure must be actionable, got {message}"
        );
    }

    /// A per-tier component with NO tier in scope is refused rather than guessed. Picking the first
    /// matching row would silently serve a bf16 text encoder to a q4 render.
    #[test]
    fn mage_co_requisites_refuse_to_guess_a_tier_when_none_is_resolved() {
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let manifest = builtin_manifest_entry("mage_flow_base");
        stage_mage_components(data_dir.path(), &manifest, "q4");

        let error = resolve_co_requisites_for_tier(
            &mage_descriptor("mage_flow_base"),
            &manifest,
            &settings_at(data_dir.path().to_path_buf()),
            None,
        )
        .expect_err("three per-tier rows with no tier in scope must not be guessed");
        assert!(
            format!("{error:?}").contains("refusing to guess"),
            "got {error:?}"
        );
    }

    #[test]
    fn moss_codec_co_requisite_resolves_from_every_live_moss_tts_entry() {
        for &(model_id, family, repo, revision, file) in MOSS_CODEC_SNAPSHOTS {
            let _env = crate::test_env::isolate_hf_cache();
            let data_dir = tempfile::tempdir().expect("temp data dir");
            let manifest = builtin_manifest_entry(model_id);
            let codec_download = manifest["downloads"]
                .as_array()
                .expect("downloads array")
                .iter()
                .find(|download| {
                    download.get("componentId").and_then(Value::as_str) == Some("codec")
                })
                .expect("codec co-requisite");
            let declared_files: Vec<&str> = codec_download["files"]
                .as_array()
                .expect("codec files")
                .iter()
                .filter_map(Value::as_str)
                .collect();
            assert!(
                declared_files.contains(&file),
                "{model_id}: fixture anchor must remain in the live codec file set"
            );
            for declared in declared_files {
                stage_snapshot_file(data_dir.path(), repo, revision, declared);
            }

            let components = resolve_co_requisites(
                &moss_descriptor(model_id, family),
                &manifest,
                &settings_at(data_dir.path().to_path_buf()),
            )
            .unwrap_or_else(|error| {
                panic!("{model_id}: the codec is staged, so resolution must succeed: {error:?}")
            });

            // The single key is EXACTLY the descriptor's advertised `codec` id — matched on the
            // manifest's explicit `componentId`, never inferred from the repo name (sc-13679).
            let keys: Vec<&str> = components.keys().map(String::as_str).collect();
            assert_eq!(
                keys,
                vec!["codec"],
                "{model_id}: the component map must carry exactly the `codec` id"
            );
            assert!(
                components.contains_key("codec"),
                "{model_id}: the `codec` component resolves to its staged snapshot"
            );
        }
    }

    #[test]
    fn a_missing_moss_codec_fails_with_an_actionable_error_naming_codec_and_repo() {
        for &(model_id, family, repo, _revision, _file) in MOSS_CODEC_SNAPSHOTS {
            let _env = crate::test_env::isolate_hf_cache();
            let data_dir = tempfile::tempdir().expect("temp data dir");
            // The codec snapshot is ABSENT — install-state gates on the hard codec coRequisite, so the
            // job must fail HERE (before the engine load), not with a mid-render hub fetch.
            let error = resolve_co_requisites(
                &moss_descriptor(model_id, family),
                &builtin_manifest_entry(model_id),
                &settings_at(data_dir.path().to_path_buf()),
            )
            .expect_err("a missing codec must fail the job before the engine load");

            let WorkerError::InvalidPayload(message) = error else {
                panic!("{model_id}: a missing codec must surface as InvalidPayload, got {error:?}");
            };
            assert!(
                message.contains("codec") && message.contains(repo),
                "{model_id}: the error must name the `codec` component + its repo BEFORE engine load, \
                 got: {message}"
            );
        }
    }

    // --- MMAudio video→audio 5-component assembly (epic 13678, sc-13684) ------------------------------

    /// A candle MMAudio generator descriptor shape at the inference pin: it advertises the FIVE
    /// caller-staged components mmaudio assembles (`candle-audio-mmaudio` `generator::REQUIRED_COMPONENTS`
    /// / `generator_44k` both declare `required_components: &["clip", "synchformer", "dit", "vae",
    /// "vocoder"]`). mmaudio is a PURE assembly — `spec.weights` is unused — so ALL FIVE must resolve
    /// from `componentId`-tagged coRequisites via the generic `resolve_co_requisites` seam.
    fn mmaudio_descriptor(id: &'static str) -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id,
            family: "mmaudio",
            backend: "candle",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &["clip", "synchformer", "dit", "vae", "vocoder"],
            control_kinds: None,
        }
    }

    /// `(model_id, [(componentId, repo, pinned revision, file)])` — the exact five component snapshots
    /// each SHIPPED MMAudio tier pins (sc-13684). Binds these tests to the LIVE `builtin.models.jsonc`,
    /// so a dropped component coRequisite or a `componentId` that stops matching the descriptor's
    /// `required_components` fails HERE, not silently at a future video→audio render. The 44k tier shares
    /// `clip` + `synchformer`, has its own `dit`/`vae`, and swaps the in-repo 16k BigVGAN `vocoder` for
    /// the external NVIDIA BigVGAN v2 repo.
    #[allow(clippy::type_complexity)]
    fn mmaudio_component_snapshots() -> Vec<(
        &'static str,
        Vec<(&'static str, &'static str, &'static str, &'static str)>,
    )> {
        const MMAUDIO_REV: &str = "eb13a1a98fdbec91753775c57b074ccdfc60587c";
        const CLIP: (&str, &str, &str, &str) = (
            "clip",
            "apple/DFN5B-CLIP-ViT-H-14-378",
            "01b771ed0d1395ca5ffdd279897d665ebe00dfd2",
            "open_clip_pytorch_model.bin",
        );
        let synchformer = (
            "synchformer",
            "hkchengrex/MMAudio",
            MMAUDIO_REV,
            "ext_weights/synchformer_state_dict.pth",
        );
        vec![
            (
                "mmaudio_small_16k",
                vec![
                    CLIP,
                    synchformer,
                    (
                        "dit",
                        "hkchengrex/MMAudio",
                        MMAUDIO_REV,
                        "weights/mmaudio_small_16k.pth",
                    ),
                    (
                        "vae",
                        "hkchengrex/MMAudio",
                        MMAUDIO_REV,
                        "ext_weights/v1-16.pth",
                    ),
                    (
                        "vocoder",
                        "hkchengrex/MMAudio",
                        MMAUDIO_REV,
                        "ext_weights/best_netG.pt",
                    ),
                ],
            ),
            (
                "mmaudio_large_44k",
                vec![
                    CLIP,
                    synchformer,
                    (
                        "dit",
                        "hkchengrex/MMAudio",
                        MMAUDIO_REV,
                        "weights/mmaudio_large_44k_v2.pth",
                    ),
                    (
                        "vae",
                        "hkchengrex/MMAudio",
                        MMAUDIO_REV,
                        "ext_weights/v1-44.pth",
                    ),
                    (
                        "vocoder",
                        "nvidia/bigvgan_v2_44khz_128band_512x",
                        "95a9d1dcb12906c03edd938d77b9333d6ded7dfb",
                        "bigvgan_generator.pt",
                    ),
                ],
            ),
        ]
    }

    #[test]
    fn mmaudio_co_requisites_resolve_all_five_components_from_every_live_tier() {
        for (model_id, components) in mmaudio_component_snapshots() {
            let _env = crate::test_env::isolate_hf_cache();
            let data_dir = tempfile::tempdir().expect("temp data dir");
            for (_id, repo, revision, file) in &components {
                stage_snapshot_file(data_dir.path(), repo, revision, file);
            }

            let resolved = resolve_co_requisites(
                &mmaudio_descriptor(model_id),
                &builtin_manifest_entry(model_id),
                &settings_at(data_dir.path().to_path_buf()),
            )
            .unwrap_or_else(|error| {
                panic!("{model_id}: all five components are staged, so resolution must succeed: {error:?}")
            });

            // The map keys are EXACTLY the descriptor's five advertised ids (order-independent BTreeMap).
            let mut keys: Vec<&str> = resolved.keys().map(String::as_str).collect();
            keys.sort_unstable();
            let mut want = mmaudio_descriptor(model_id).required_components.to_vec();
            want.sort_unstable();
            assert_eq!(
                keys, want,
                "{model_id}: the component map keys must match the descriptor's required_components"
            );
            // Each single-file component resolves to a `File` at the staged snapshot path — the exact
            // file the tier's `componentId` download pins (matched on `componentId`, never repo name).
            for (id, _repo, _revision, file) in &components {
                match resolved.get(*id) {
                    Some(gen_core::WeightsSource::File(path)) => assert!(
                        path.ends_with(file),
                        "{model_id}: component `{id}` must resolve to {file}, got {path:?}"
                    ),
                    other => {
                        panic!("{model_id}: component `{id}` must resolve to a File, got {other:?}")
                    }
                }
            }
        }
    }

    #[test]
    fn mmaudio_clip_repo_rename_relinks_the_existing_pinned_cache_without_a_fetch() {
        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        let (model_id, components) = mmaudio_component_snapshots()
            .into_iter()
            .next()
            .expect("MMAudio tier fixture");
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let (_, canonical_clip_repo, clip_revision, clip_file) = *components
            .iter()
            .find(|(id, _, _, _)| *id == "clip")
            .expect("clip component present");
        let legacy_clip = stage_snapshot_file(
            data_dir.path(),
            rename.legacy_repo,
            clip_revision,
            clip_file,
        );
        for (id, repo, revision, file) in &components {
            if *id != "clip" {
                stage_snapshot_file(data_dir.path(), repo, revision, file);
            }
        }

        let resolved = resolve_co_requisites(
            &mmaudio_descriptor(model_id),
            &builtin_manifest_entry(model_id),
            &settings_at(data_dir.path().to_path_buf()),
        )
        .expect("the canonical manifest re-links the exact legacy pinned CLIP cache");
        let gen_core::WeightsSource::File(canonical_clip) =
            resolved.get("clip").expect("resolved CLIP component")
        else {
            panic!("the CLIP component must resolve to a single file");
        };
        let expected = huggingface_repo_cache_path(data_dir.path(), canonical_clip_repo)
            .expect("canonical repo cache")
            .join("snapshots")
            .join(clip_revision)
            .join(clip_file);
        assert_eq!(
            canonical_clip,
            &std::fs::canonicalize(&expected).expect("canonical linked file")
        );
        assert_eq!(
            std::fs::read(canonical_clip).expect("read canonical link"),
            b"weights"
        );
        assert!(legacy_clip.is_file(), "the source cache remains intact");
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            assert_eq!(
                std::fs::metadata(&legacy_clip)
                    .expect("legacy metadata")
                    .ino(),
                std::fs::metadata(canonical_clip)
                    .expect("canonical metadata")
                    .ino(),
                "the corrected repo must hard-link the existing blob, not copy or fetch it"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn renamed_current_snapshot_symlink_resolves_to_its_validated_plain_blob() {
        use std::os::unix::fs::symlink;

        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        let (model_id, components) = mmaudio_component_snapshots()
            .into_iter()
            .next()
            .expect("MMAudio tier fixture");
        let _env = crate::test_env::isolate_hf_cache();
        let data_dir = tempfile::tempdir().expect("temp data dir");
        let current_repo =
            huggingface_repo_cache_path(data_dir.path(), rename.current_repo).unwrap();
        let current_snapshot = current_repo.join("snapshots").join(rename.revision);
        let current_blob = current_repo.join("blobs").join("clip-blob");
        std::fs::create_dir_all(&current_snapshot).unwrap();
        std::fs::create_dir_all(current_blob.parent().unwrap()).unwrap();
        std::fs::write(&current_blob, b"canonical blob bytes").unwrap();
        symlink(&current_blob, current_snapshot.join(rename.files[0])).unwrap();
        for (id, repo, revision, file) in &components {
            if *id != "clip" {
                stage_snapshot_file(data_dir.path(), repo, revision, file);
            }
        }

        let resolved = resolve_co_requisites(
            &mmaudio_descriptor(model_id),
            &builtin_manifest_entry(model_id),
            &settings_at(data_dir.path().to_path_buf()),
        )
        .expect("the exact current cache resolves without traversing its snapshot symlink");
        let gen_core::WeightsSource::File(resolved_clip) = &resolved["clip"] else {
            panic!("the CLIP component must resolve to a single file");
        };
        assert_eq!(
            resolved_clip,
            &std::fs::canonicalize(&current_blob).unwrap()
        );
        assert_eq!(
            std::fs::read(resolved_clip).unwrap(),
            b"canonical blob bytes"
        );
        assert!(
            !huggingface_repo_cache_path(data_dir.path(), rename.legacy_repo)
                .unwrap()
                .exists(),
            "a valid current cache must not consult or create the legacy namespace"
        );
    }

    #[test]
    fn renamed_snapshot_migration_uses_validated_legacy_when_hardlinks_are_unavailable() {
        let _env = crate::test_env::isolate_hf_cache();
        let temp = tempfile::tempdir().expect("temporary cache");
        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        let legacy_file = stage_snapshot_file(
            temp.path(),
            rename.legacy_repo,
            rename.revision,
            rename.files[0],
        );
        let files = rename
            .files
            .iter()
            .map(|file| (*file).to_owned())
            .collect::<Vec<_>>();

        let resolved = resolve_renamed_hf_single_file_using(
            temp.path(),
            rename.current_repo,
            rename.revision,
            &files,
            |_, _| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "fixture filesystem has no hardlinks",
                ))
            },
        )
        .expect("an unsupported hardlink must fall back to the validated legacy file");

        let canonical_file = huggingface_repo_cache_path(temp.path(), rename.current_repo)
            .unwrap()
            .join("snapshots")
            .join(rename.revision)
            .join(rename.files[0]);
        assert_eq!(
            resolved,
            std::fs::canonicalize(&legacy_file).unwrap(),
            "a no-hardlink filesystem must not duplicate a multi-GB immutable blob"
        );
        assert!(
            !canonical_file.exists(),
            "the direct legacy fallback must not copy the blob into the canonical namespace"
        );
        assert_eq!(std::fs::read(&legacy_file).unwrap(), b"weights");
    }

    #[cfg(unix)]
    #[test]
    fn renamed_single_file_symlink_source_bypasses_reparse_when_hardlinks_are_unavailable() {
        use std::os::unix::fs::symlink;

        let _env = crate::test_env::isolate_hf_cache();
        let temp = tempfile::tempdir().expect("temporary cache");
        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        let legacy_repo = huggingface_repo_cache_path(temp.path(), rename.legacy_repo).unwrap();
        let legacy_snapshot = legacy_repo.join("snapshots").join(rename.revision);
        let legacy_blob = legacy_repo.join("blobs").join("clip-blob");
        std::fs::create_dir_all(&legacy_snapshot).unwrap();
        std::fs::create_dir_all(legacy_blob.parent().unwrap()).unwrap();
        std::fs::write(&legacy_blob, b"validated blob bytes").unwrap();
        symlink(&legacy_blob, legacy_snapshot.join(rename.files[0])).unwrap();
        let files = rename
            .files
            .iter()
            .map(|file| (*file).to_owned())
            .collect::<Vec<_>>();

        let resolved = resolve_renamed_hf_single_file_using(
            temp.path(),
            rename.current_repo,
            rename.revision,
            &files,
            |_, _| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "fixture filesystem has no hardlinks",
                ))
            },
        )
        .expect("the validated plain blob is the no-hardlink fallback");

        assert_eq!(resolved, std::fs::canonicalize(&legacy_blob).unwrap());
        assert_eq!(std::fs::read(&resolved).unwrap(), b"validated blob bytes");
        assert!(
            std::fs::symlink_metadata(legacy_snapshot.join(rename.files[0]))
                .unwrap()
                .file_type()
                .is_symlink(),
            "the exact rename seam must not rewrite the legacy snapshot symlink"
        );
        assert!(
            !huggingface_repo_cache_path(temp.path(), rename.current_repo)
                .unwrap()
                .join("snapshots")
                .join(rename.revision)
                .join(rename.files[0])
                .exists(),
            "the fallback must not copy the blob into the canonical namespace"
        );

        // Exercise the actual co-requisite resolver through the same fallback. Poisoning only the
        // canonical snapshot directory makes materialization unavailable while the validated
        // legacy blob remains exactly the cache state the API reports as installed.
        let current_repo = huggingface_repo_cache_path(temp.path(), rename.current_repo).unwrap();
        std::fs::remove_dir_all(&current_repo).unwrap();
        std::fs::create_dir_all(&current_repo).unwrap();
        let outside_snapshots = temp.path().join("outside-snapshots");
        std::fs::create_dir_all(&outside_snapshots).unwrap();
        symlink(&outside_snapshots, current_repo.join("snapshots")).unwrap();
        let (model_id, components) = mmaudio_component_snapshots()
            .into_iter()
            .next()
            .expect("MMAudio tier fixture");
        for (id, repo, revision, file) in &components {
            if *id != "clip" {
                stage_snapshot_file(temp.path(), repo, revision, file);
            }
        }
        let resolved_components = resolve_co_requisites(
            &mmaudio_descriptor(model_id),
            &builtin_manifest_entry(model_id),
            &settings_at(temp.path().to_path_buf()),
        )
        .expect("production resolution uses the validated plain legacy blob");
        let gen_core::WeightsSource::File(resolved_clip) = &resolved_components["clip"] else {
            panic!("the CLIP component must resolve to a single file");
        };
        assert_eq!(
            resolved_clip,
            &std::fs::canonicalize(&legacy_blob).unwrap(),
            "production must return the validator-proven plain blob path, not its snapshot symlink"
        );
        assert_eq!(
            std::fs::read(resolved_clip).unwrap(),
            b"validated blob bytes"
        );
        assert!(
            !outside_snapshots.join(rename.revision).exists(),
            "production must not create through the poisoned canonical snapshot directory"
        );
    }

    #[test]
    fn renamed_snapshot_migration_is_race_safe_and_rejects_wrong_revision() {
        let _env = crate::test_env::isolate_hf_cache();
        let temp = tempfile::tempdir().expect("temporary cache");
        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        stage_snapshot_file(
            temp.path(),
            rename.legacy_repo,
            rename.revision,
            rename.files[0],
        );
        let files = rename
            .files
            .iter()
            .map(|file| (*file).to_owned())
            .collect::<Vec<_>>();

        std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                resolve_renamed_hf_single_file(
                    temp.path(),
                    rename.current_repo,
                    rename.revision,
                    &files,
                )
            });
            let second = scope.spawn(|| {
                resolve_renamed_hf_single_file(
                    temp.path(),
                    rename.current_repo,
                    rename.revision,
                    &files,
                )
            });
            assert!(first.join().unwrap().is_some());
            assert!(second.join().unwrap().is_some());
        });
        assert!(resolve_renamed_hf_single_file(
            temp.path(),
            rename.current_repo,
            "wrong-revision",
            &files,
        )
        .is_none());
    }

    #[cfg(unix)]
    #[test]
    fn renamed_snapshot_migration_rejects_a_legacy_file_escaping_its_repo_cache() {
        use std::os::unix::fs::symlink;

        let _env = crate::test_env::isolate_hf_cache();
        let temp = tempfile::tempdir().expect("temporary cache");
        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        let legacy_snapshot = huggingface_repo_cache_path(temp.path(), rename.legacy_repo)
            .unwrap()
            .join("snapshots")
            .join(rename.revision);
        std::fs::create_dir_all(&legacy_snapshot).unwrap();
        let outside = temp.path().join("outside.bin");
        std::fs::write(&outside, b"not cache-owned").unwrap();
        symlink(&outside, legacy_snapshot.join(rename.files[0])).unwrap();
        let files = rename
            .files
            .iter()
            .map(|file| (*file).to_owned())
            .collect::<Vec<_>>();

        assert!(resolve_renamed_hf_single_file(
            temp.path(),
            rename.current_repo,
            rename.revision,
            &files,
        )
        .is_none());
    }

    #[cfg(unix)]
    #[test]
    fn renamed_snapshot_migration_falls_back_to_validated_legacy_for_destination_escapes() {
        use std::os::unix::fs::symlink;

        let _env = crate::test_env::isolate_hf_cache();
        let temp = tempfile::tempdir().expect("temporary cache");
        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        let legacy_file = stage_snapshot_file(
            temp.path(),
            rename.legacy_repo,
            rename.revision,
            rename.files[0],
        );
        let files = rename
            .files
            .iter()
            .map(|file| (*file).to_owned())
            .collect::<Vec<_>>();
        let current_repo = huggingface_repo_cache_path(temp.path(), rename.current_repo).unwrap();
        std::fs::create_dir_all(&current_repo).unwrap();

        let outside_snapshots = temp.path().join("outside-snapshots");
        std::fs::create_dir_all(&outside_snapshots).unwrap();
        symlink(&outside_snapshots, current_repo.join("snapshots")).unwrap();
        let resolved_file = resolve_renamed_hf_single_file(
            temp.path(),
            rename.current_repo,
            rename.revision,
            &files,
        )
        .expect("a poisoned canonical directory must fall back to the valid legacy file");
        assert_eq!(resolved_file, std::fs::canonicalize(&legacy_file).unwrap());
        assert!(
            !outside_snapshots.join(rename.revision).exists(),
            "validation must happen before creating a directory through an escaping parent"
        );

        std::fs::remove_file(current_repo.join("snapshots")).unwrap();
        let current_snapshot = current_repo.join("snapshots").join(rename.revision);
        std::fs::create_dir_all(&current_snapshot).unwrap();
        let outside_file = temp.path().join("outside-destination.bin");
        std::fs::write(&outside_file, b"outside must not be accepted").unwrap();
        symlink(&outside_file, current_snapshot.join(rename.files[0])).unwrap();
        let resolved_file = resolve_renamed_hf_single_file(
            temp.path(),
            rename.current_repo,
            rename.revision,
            &files,
        )
        .expect("a poisoned canonical file must fall back to the valid legacy file");
        assert_eq!(resolved_file, std::fs::canonicalize(&legacy_file).unwrap());
        assert_eq!(
            std::fs::read(&outside_file).unwrap(),
            b"outside must not be accepted"
        );

        let (model_id, components) = mmaudio_component_snapshots()
            .into_iter()
            .next()
            .expect("MMAudio tier fixture");
        for (id, repo, revision, file) in &components {
            if *id != "clip" {
                stage_snapshot_file(temp.path(), repo, revision, file);
            }
        }
        let resolved = resolve_co_requisites(
            &mmaudio_descriptor(model_id),
            &builtin_manifest_entry(model_id),
            &settings_at(temp.path().to_path_buf()),
        )
        .expect("production must use the validated legacy snapshot without following the poison");
        let gen_core::WeightsSource::File(resolved_clip) =
            resolved.get("clip").expect("resolved CLIP component")
        else {
            panic!("the CLIP component must resolve to a single file");
        };
        assert_eq!(
            std::fs::canonicalize(resolved_clip).unwrap(),
            std::fs::canonicalize(&legacy_file).unwrap(),
            "the worker must load the confined legacy file that made the catalog ready"
        );
        assert_eq!(
            std::fs::read(&outside_file).unwrap(),
            b"outside must not be accepted",
            "the poisoned canonical destination remains untouched"
        );
    }

    #[test]
    fn a_missing_mmaudio_component_fails_with_an_actionable_error_naming_id_and_repo() {
        for (model_id, components) in mmaudio_component_snapshots() {
            let _env = crate::test_env::isolate_hf_cache();
            let data_dir = tempfile::tempdir().expect("temp data dir");
            // Stage every component EXCEPT the vocoder, whose snapshot is ABSENT — install-state gates on
            // the hard component coRequisites, so the job must fail HERE (before the engine load), naming
            // the missing id + its repo, not with a mid-render hub fetch.
            let (_vid, vocoder_repo, _vrev, _vfile) = *components
                .iter()
                .find(|(id, _, _, _)| *id == "vocoder")
                .expect("vocoder component present");
            for (id, repo, revision, file) in &components {
                if *id == "vocoder" {
                    continue;
                }
                stage_snapshot_file(data_dir.path(), repo, revision, file);
            }

            let error = resolve_co_requisites(
                &mmaudio_descriptor(model_id),
                &builtin_manifest_entry(model_id),
                &settings_at(data_dir.path().to_path_buf()),
            )
            .expect_err("a missing mmaudio component must fail the job before the engine load");

            let WorkerError::InvalidPayload(message) = error else {
                panic!(
                    "{model_id}: a missing component must surface as InvalidPayload, got {error:?}"
                );
            };
            assert!(
                message.contains("vocoder") && message.contains(vocoder_repo),
                "{model_id}: the error must name the missing `vocoder` component + its repo BEFORE engine \
                 load, got: {message}"
            );
        }
    }

    /// sc-25909 — a v0.9.0 install of the Qwen-Image 2.1 re-host (pinned at the superseded
    /// `1691de01…`) is carried to the republished `ef03535e…` cache-only: every carried entry is a
    /// relative link to the SAME legacy blob (no copy, no fetch), the added NOTICE is written from
    /// its declared bytes as its own git-oid blob, and only the requested tier is touched.
    ///
    /// *Mutation that reds this:* `huggingface_pinned_tier_snapshot_dir` skipping
    /// `materialize_superseded_hf_revision` (the tier stays stranded at the old revision).
    #[cfg(unix)]
    #[test]
    fn superseded_revision_migration_builds_the_new_snapshot_from_the_legacy_cache() {
        use std::os::unix::fs::MetadataExt as _;
        let _env = crate::test_env::isolate_hf_cache();
        let data = tempfile::tempdir().expect("temp data dir");
        let supersession =
            sceneworks_core::hf_revision_supersessions::QWEN_IMAGE_2_1_Q8_NOTICE_SUPERSESSION;
        let legacy = stage_superseded_legacy_tier_for_test(data.path(), &supersession, true, &[]);
        let files = vec!["q8/*".to_owned()];
        assert!(
            huggingface_pinned_snapshot_dir(
                data.path(),
                supersession.repo,
                supersession.current_revision
            )
            .is_none(),
            "precondition: nothing is cached at the current pin"
        );

        let snapshot = huggingface_pinned_tier_snapshot_dir(
            data.path(),
            supersession.repo,
            supersession.current_revision,
            &files,
        )
        .expect("the superseded install is carried to the current pin");
        assert!(snapshot.ends_with(supersession.current_revision));
        for file in supersession.carried {
            let entry = snapshot.join(file.path);
            assert_eq!(
                std::fs::read_link(&entry).expect("carried entry is a snapshot symlink"),
                std::fs::read_link(legacy.join(file.path)).expect("legacy symlink"),
                "{}: links the same relative blob the legacy snapshot does",
                file.path
            );
            assert_eq!(
                std::fs::metadata(&entry).unwrap().ino(),
                std::fs::metadata(legacy.join(file.path)).unwrap().ino(),
                "{}: no copy",
                file.path
            );
        }
        let notice = snapshot.join("q8/NOTICE");
        assert_eq!(
            std::fs::read(&notice).unwrap(),
            supersession.added[0].contents
        );
        assert_eq!(
            std::fs::read_link(&notice).unwrap(),
            Path::new("../../../blobs").join(supersession.added[0].blob)
        );
        assert!(
            !snapshot.join("q4").exists(),
            "only the requested tier is materialized"
        );
        assert!(
            legacy.join("q8/model_index.json").is_file(),
            "the legacy snapshot is intact"
        );
    }

    /// Re-running is a no-op on a complete tier, and an interrupted run (entries missing from the
    /// new snapshot) is completed rather than duplicated.
    ///
    /// *Mutation that reds this:* `link_superseded_snapshot_entry` replacing an existing entry, or
    /// `write_declared_blob` rejecting the blob it already wrote.
    #[cfg(unix)]
    #[test]
    fn superseded_revision_migration_is_idempotent_and_completes_an_interrupted_run() {
        let _env = crate::test_env::isolate_hf_cache();
        let data = tempfile::tempdir().expect("temp data dir");
        let supersession =
            sceneworks_core::hf_revision_supersessions::QWEN_IMAGE_2_1_Q4_NOTICE_SUPERSESSION;
        stage_superseded_legacy_tier_for_test(data.path(), &supersession, true, &[]);
        let files = vec!["q4/*".to_owned()];
        let resolve = || {
            huggingface_pinned_tier_snapshot_dir(
                data.path(),
                supersession.repo,
                supersession.current_revision,
                &files,
            )
            .expect("resolves")
        };
        let first = resolve();
        let blobs = huggingface_repo_cache_path(data.path(), supersession.repo)
            .unwrap()
            .join("blobs");
        let blob_count = || std::fs::read_dir(&blobs).unwrap().count();
        let before = blob_count();
        let link_before = std::fs::read_link(first.join("q4/vae/config.json")).unwrap();
        assert_eq!(resolve(), first);
        assert_eq!(
            blob_count(),
            before,
            "no second NOTICE blob, no temp left behind"
        );
        assert_eq!(
            std::fs::read_link(first.join("q4/vae/config.json")).unwrap(),
            link_before
        );

        std::fs::remove_file(first.join("q4/NOTICE")).unwrap();
        std::fs::remove_file(first.join("q4/transformer/model.safetensors")).unwrap();
        assert_eq!(resolve(), first);
        assert!(
            sceneworks_core::hf_revision_supersessions::validate_superseding_current_snapshot(
                &huggingface_hub_cache_dir(data.path()),
                &huggingface_repo_cache_path(data.path(), supersession.repo).unwrap(),
                &first,
                &supersession,
            )
            .is_some()
        );
        assert_eq!(blob_count(), before);
    }

    /// A legacy tier missing any carried file — or holding one at the wrong size (a torn blob) — is
    /// refused, and nothing is written at the current pin.
    ///
    /// *Mutation that reds this:* `validate_superseded_legacy_snapshot` accepting a subset (or
    /// dropping the size check).
    #[cfg(unix)]
    #[test]
    fn superseded_revision_migration_refuses_a_partial_legacy_snapshot() {
        let _env = crate::test_env::isolate_hf_cache();
        let supersession =
            sceneworks_core::hf_revision_supersessions::QWEN_IMAGE_2_1_Q8_NOTICE_SUPERSESSION;
        let files = vec!["q8/*".to_owned()];
        for damage in ["missing", "truncated"] {
            let data = tempfile::tempdir().expect("temp data dir");
            let legacy =
                stage_superseded_legacy_tier_for_test(data.path(), &supersession, true, &[]);
            let victim = legacy.join("q8/processor/vocab.json");
            match damage {
                "missing" => std::fs::remove_file(&victim).unwrap(),
                _ => std::fs::OpenOptions::new()
                    .write(true)
                    .open(&victim)
                    .unwrap()
                    .set_len(10)
                    .unwrap(),
            }
            assert!(
                huggingface_pinned_tier_snapshot_dir(
                    data.path(),
                    supersession.repo,
                    supersession.current_revision,
                    &files,
                )
                .is_none(),
                "{damage}: a partial legacy tier is not carried forward"
            );
            assert!(
                !huggingface_repo_cache_path(data.path(), supersession.repo)
                    .unwrap()
                    .join("snapshots")
                    .join(supersession.current_revision)
                    .exists(),
                "{damage}: nothing is written at the current pin"
            );
        }
    }

    /// A legacy cache whose entries are plain files (the Windows hardlink layout our downloader
    /// writes) is carried by hard link, never by copy.
    #[cfg(unix)]
    #[test]
    fn superseded_revision_migration_hard_links_plain_legacy_entries() {
        use std::os::unix::fs::MetadataExt as _;
        let _env = crate::test_env::isolate_hf_cache();
        let data = tempfile::tempdir().expect("temp data dir");
        let supersession =
            sceneworks_core::hf_revision_supersessions::QWEN_IMAGE_2_1_Q8_NOTICE_SUPERSESSION;
        let legacy = stage_superseded_legacy_tier_for_test(data.path(), &supersession, false, &[]);
        let snapshot = huggingface_pinned_tier_snapshot_dir(
            data.path(),
            supersession.repo,
            supersession.current_revision,
            &["q8/*".to_owned()],
        )
        .expect("resolves");
        let entry = snapshot.join("q8/text_encoder/model.safetensors");
        assert!(!std::fs::symlink_metadata(&entry)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(
            std::fs::metadata(&entry).unwrap().ino(),
            std::fs::metadata(legacy.join("q8/text_encoder/model.safetensors"))
                .unwrap()
                .ino()
        );
    }
}

/// The linked half of E8: a LINKED copy of an NVFP4 checkpoint is exposed exactly as a managed one
/// (sc-11045, epic 11037).
///
/// Ungated on purpose. The stamp and the resolvers that read it are platform-independent — the
/// backend-specific dispatch half lives in `image_jobs::tests` — and this is the layer where "the
/// same bytes, owned differently, are the same checkpoint" is decided.
#[cfg(test)]
mod linked_classification_tests {
    use super::*;
    use sceneworks_core::checkpoint_weight_facts::NVFP4_CODEC_ID;
    use std::path::PathBuf;

    fn write_header_file(
        path: &Path,
        metadata: Value,
        entries: &[(String, &'static str, Vec<u64>)],
    ) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), metadata);
        let mut offset = 0_u64;
        for (name, dtype, shape) in entries {
            let width = match *dtype {
                "F32" => 4,
                "BF16" | "F16" => 2,
                _ => 1,
            };
            let bytes = shape.iter().product::<u64>().max(1) * width;
            header.insert(
                name.clone(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.resize(bytes.len() + offset as usize, 0x5a);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// A single-file NVFP4 transformer whose header carries everything the classifier proves: the
    /// `_quantization_metadata` layer map, and one complete packed triplet per projection.
    fn write_nvfp4_transformer(path: &Path) {
        let layers = 6;
        let mut declared = serde_json::Map::new();
        for index in 0..layers {
            declared.insert(
                format!("blocks.{index}.attn.wq"),
                json!({"format": "nvfp4", "group_size": 16}),
            );
        }
        let mut entries: Vec<(String, &'static str, Vec<u64>)> = vec![(
            "model.diffusion_model.txtfusion.projector.weight".to_owned(),
            "BF16",
            vec![128, 128],
        )];
        for index in 0..layers {
            let base = format!("blocks.{index}.attn.wq");
            entries.push((format!("{base}.weight"), "U8", vec![128, 32]));
            entries.push((format!("{base}.weight_scale"), "F8_E4M3", vec![128, 4]));
            entries.push((format!("{base}.weight_scale_2"), "F32", vec![]));
        }
        write_header_file(
            path,
            json!({
                "format": "pt",
                "_quantization_metadata": serde_json::to_string(&json!({"layers": declared}))
                    .unwrap(),
            }),
            &entries,
        );
    }

    /// The same family surface, stored DENSE — the control row.
    fn write_dense_transformer(path: &Path) {
        write_header_file(
            path,
            json!({ "format": "pt" }),
            &[
                (
                    "model.diffusion_model.txtfusion.projector.weight".to_owned(),
                    "BF16",
                    vec![128, 128],
                ),
                (
                    "model.diffusion_model.blocks.0.attn.wq.weight".to_owned(),
                    "BF16",
                    vec![128, 128],
                ),
                (
                    "model.diffusion_model.first.weight".to_owned(),
                    "BF16",
                    vec![128, 128],
                ),
            ],
        )
    }

    struct LinkedFixture {
        _data: tempfile::TempDir,
        _library: tempfile::TempDir,
        store: CheckpointPlanStore,
        library_dir: PathBuf,
        root_id: String,
    }

    fn linked_fixture(label: &str) -> LinkedFixture {
        let data = tempfile::Builder::new()
            .prefix(&format!("linked-e8-{label}-data-{}-", std::process::id()))
            .tempdir()
            .unwrap();
        let library = tempfile::Builder::new()
            .prefix(&format!("linked-e8-{label}-lib-{}-", std::process::id()))
            .tempdir()
            .unwrap();
        let library_dir = std::fs::canonicalize(library.path()).unwrap();
        let store = CheckpointPlanStore::open(data.path());
        let root_id = store.approve_root(&library_dir).unwrap().root_id;
        LinkedFixture {
            _data: data,
            _library: library,
            store,
            library_dir,
            root_id,
        }
    }

    /// The queued entry the API hands the worker, before any stamp.
    fn queued_entry(root_id: &str, relative_path: &str) -> JsonObject {
        json!({
            "id": "linked_kreamania_nvfp4",
            "name": "kreamania nvfp4",
            "type": "image",
            "catalogScope": "user",
            "source": {
                "provider": "linked-library",
                "rootId": root_id,
                "relativePath": relative_path,
            },
        })
        .as_object()
        .unwrap()
        .clone()
    }

    /// A LINKED NVFP4 checkpoint resolves to the `nvfp4-v1` SOURCE CODEC, exactly as a managed copy
    /// of the identical bytes does.
    ///
    /// This is the one fact every E8 surface reads: admission's native claim, the catalog's
    /// `runtimeQuantTiers`, the telemetry `source_codec`, the asset receipt's facts object, and the
    /// worker's dispatch guard. Before the stamp it answered `None` for a linked entry, so all five
    /// were silently absent for a checkpoint that is stored packed.
    ///
    /// Failing mutation (run): delete the `if let Some((source_shape, quant_format)) =
    /// classification` block in `linked_checkpoint_manifest_entry`.
    #[test]
    fn a_linked_nvfp4_checkpoint_is_stamped_and_resolves_to_the_nvfp4_source_codec() {
        let fx = linked_fixture("codec");
        let relative = "kreamania_nvfp4.safetensors";
        let file = fx.library_dir.join(relative);
        write_nvfp4_transformer(&file);
        let compiled = fx.store.compile_linked(&fx.root_id, relative).unwrap();

        let classification = base_weight_manifest_classification(&file);
        assert_eq!(
            classification,
            Some(("transformer_file", "nvfp4")),
            "the header proves a packed single-file transformer"
        );

        let entry = linked_checkpoint_manifest_entry(
            queued_entry(&fx.root_id, relative),
            &compiled,
            classification,
        );

        assert_eq!(
            entry.get("importQuantFormat").and_then(Value::as_str),
            Some("nvfp4")
        );
        assert_eq!(
            entry.get("importSourceShape").and_then(Value::as_str),
            Some("transformer_file"),
            "without a source shape the catalog's provider surface early-returns and emits no \
             runtimeQuantTiers at all"
        );
        assert_eq!(
            sceneworks_core::jobs_store::imported_entry_source_codec(&entry),
            Some(NVFP4_CODEC_ID),
            "the linked entry answers the SAME source-codec question a managed one does"
        );
        // The linked invariants sc-20635 established are untouched.
        assert_eq!(
            sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(&entry),
            Some(compiled.checkpoint_id.as_str())
        );
        assert!(
            sceneworks_core::jobs_store::imported_entry_loadable_path(&entry).is_none(),
            "a linked entry still owns no installed bytes, so no bespoke lane can claim it"
        );
    }

    /// A DENSE linked checkpoint is stamped with its own classification and never with NVFP4 —
    /// the stamp is a reading of the header, not a constant.
    #[test]
    fn a_linked_dense_checkpoint_is_not_stamped_nvfp4() {
        let fx = linked_fixture("dense");
        let relative = "kreamania_dense.safetensors";
        let file = fx.library_dir.join(relative);
        write_dense_transformer(&file);
        let compiled = fx.store.compile_linked(&fx.root_id, relative).unwrap();

        let classification = base_weight_manifest_classification(&file);
        assert_eq!(classification, Some(("transformer_file", "bf16")));
        let entry = linked_checkpoint_manifest_entry(
            queued_entry(&fx.root_id, relative),
            &compiled,
            classification,
        );
        assert_eq!(
            sceneworks_core::jobs_store::imported_entry_source_codec(&entry),
            Some("dense-bf16-v1")
        );
    }

    /// A header that proves nothing stamps nothing — an absent key, never an invented one.
    ///
    /// Two halves, because the plan store refuses to COMPILE a file it cannot assign a component
    /// role and a family, so the unclassifiable file never reaches the stamp in production. The
    /// reachable shape is a file that compiles while the base-weight detector still declines it, so
    /// the classifier's `None` and the stamp's handling of `None` are pinned separately.
    #[test]
    fn an_unclassifiable_linked_file_stamps_neither_key() {
        let dir = tempfile::tempdir().unwrap();
        let mystery = dir.path().join("mystery.safetensors");
        write_header_file(
            &mystery,
            json!({ "format": "pt" }),
            &[("some.random.weight".to_owned(), "F32", vec![4, 4])],
        );
        assert_eq!(
            base_weight_manifest_classification(&mystery),
            None,
            "a header with no base-weight evidence proves no classification"
        );

        let fx = linked_fixture("unknown");
        let relative = "kreamania_dense.safetensors";
        write_dense_transformer(&fx.library_dir.join(relative));
        let compiled = fx.store.compile_linked(&fx.root_id, relative).unwrap();
        let entry =
            linked_checkpoint_manifest_entry(queued_entry(&fx.root_id, relative), &compiled, None);
        assert!(entry.get("importQuantFormat").is_none());
        assert!(entry.get("importSourceShape").is_none());
        assert_eq!(
            sceneworks_core::jobs_store::imported_entry_source_codec(&entry),
            None
        );
        // ...and the plan stamp is unaffected, so an unclassified linked entry is still selectable.
        assert_eq!(
            sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(&entry),
            Some(compiled.checkpoint_id.as_str())
        );
    }

    /// The linked classifier and the managed branch's inline mapping answer the same pair for one
    /// file. Two copies of one rule that can disagree about what a checkpoint is stored as is the
    /// sc-13542 resolver-drift class; this is the witness that they do not.
    ///
    /// The managed mapping is restated here from `run_model_import_job` (it is inlined inside a
    /// `fail_job`-returning `match` and cannot be called from a test), so this compares the two
    /// TRANSCRIPTIONS. A future edit to either that changes the pair turns it red.
    #[test]
    fn the_linked_classifier_matches_the_managed_branchs_mapping() {
        let dir = tempfile::tempdir().unwrap();
        for (label, write) in [
            ("nvfp4", write_nvfp4_transformer as fn(&Path)),
            ("dense", write_dense_transformer as fn(&Path)),
        ] {
            let file = dir.path().join(format!("{label}.safetensors"));
            write(&file);
            let detection = detect_base_weight_file(&file).expect("the header parses");
            let managed = match detection {
                BaseWeightDetection::Recognized(verdict) => {
                    let source = match verdict.component {
                        ComponentRole::Transformer => Some("transformer_file"),
                        ComponentRole::Checkpoint => Some("fused_checkpoint"),
                        ComponentRole::TextEncoder | ComponentRole::Vae => None,
                    };
                    source.map(|source| (source, verdict.quant.as_str()))
                }
                BaseWeightDetection::Unrecognized { .. } => None,
            };
            assert_eq!(
                base_weight_manifest_classification(&file),
                managed,
                "{label}: the linked and managed classifications must not drift"
            );
            assert!(managed.is_some(), "{label}: fixture check — it classifies");
        }
    }
}
