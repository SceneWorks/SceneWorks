use super::*;
use serde_json::json;
// Reuse the ONE crate-wide HF-cache guard (sc-13834) so these tests resolve the cache under
// their tempdir `data_dir`, never a developer's real HF cache — and serialize on the same
// `HF_ENV_LOCK` as the `crate::tests` suite (a second lock would defeat the mutual exclusion).
use crate::tests::support::isolate_hf_cache;

/// Seed a HuggingFace repo cache snapshot under `data_dir` containing `files` (repo-relative
/// paths). Mirrors the on-disk layout `model_variant_states` probes.
fn seed_cache(data_dir: &FsPath, repo: &str, files: &[&str]) {
    let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
    let snapshot = cache.join("snapshots").join("abc123");
    for file in files {
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }
}

fn quant_matrix_model(repo: &str) -> Value {
    json!({
        "id": "matrix_model",
        "downloads": [
            {
                "provider": "huggingface",
                "repo": repo,
                "variant": "q4",
                "default": true,
                "files": ["q4/*"],
                "footprint": { "diskSizeBytes": 5_000_000_000_u64 }
            },
            {
                "provider": "huggingface",
                "repo": repo,
                "variant": "q8",
                "files": ["q8/*"],
                "footprint": { "diskSizeBytes": 10_000_000_000_u64, "peakMemoryBytes": null }
            },
            {
                "provider": "huggingface",
                "repo": repo,
                "variant": "bf16",
                "files": ["bf16/*"],
                "estimatedSizeBytes": 20_000_000_000_u64
            }
        ]
    })
}

#[test]
fn tier_dependencies_cannot_be_borrowed_from_a_sibling_or_a_receipt() {
    let _env = isolate_hf_cache();
    let data = tempfile::tempdir().unwrap();
    let repo = "SceneWorks/matrix";
    let components = "SceneWorks/encoders";
    let mut model = quant_matrix_model(repo);
    for tier in ["q4", "q8", "bf16"] {
        model["downloads"].as_array_mut().unwrap().push(json!({
            "provider": "huggingface", "repo": components, "coRequisite": true,
            "variant": tier, "files": [format!("{tier}/encoder.bin")]
        }));
    }
    seed_cache(data.path(), repo, &["bf16/model.safetensors"]);
    seed_cache(
        data.path(),
        components,
        &["q4/encoder.bin", "bf16/encoder.bin"],
    );
    let state = || install_state_for(model_download_context(&model).unwrap(), &model, data.path());
    assert!(state().installed); // Also backfills the primary receipt.
    let receipt = data.path().join("models").join(safe_download_dir(repo));
    assert!(!receipt_file_sets(&receipt, repo, Some("matrix_model")).is_empty());

    std::fs::remove_file(
        huggingface_repo_cache_path(data.path(), components)
            .unwrap()
            .join("snapshots/abc123/bf16/encoder.bin"),
    )
    .unwrap();
    let broken = state();
    assert!(
        !broken.installed,
        "q4's encoder and the bf16 primary receipt cannot certify bf16"
    );
    assert!(broken.cache_incomplete);
    assert!(broken
        .missing_required_files
        .contains(&format!("{components}/bf16/encoder.bin")));
    let variants = model_variant_states(&model, data.path());
    for tier in ["q4", "q8"] {
        let variant = variants.iter().find(|v| v.variant == tier).unwrap();
        assert!(
            !variant.installed && !variant.cache_incomplete,
            "absent primary {tier}"
        );
    }

    seed_cache(data.path(), components, &["bf16/encoder.bin"]);
    assert!(state().installed);
    assert!(!state().cache_incomplete);
}

#[test]
fn managed_optional_predictor_missing_or_corrupt_keeps_dense_installed_and_repairable() {
    let _env = isolate_hf_cache();
    let data = tempfile::tempdir().unwrap();
    let repo = "SceneWorks/matrix";
    let mut model = quant_matrix_model(repo);
    let row = json!({"provider":"huggingface","repo":"test/predictor","revision":"a".repeat(40),
        "files":["predictor.safetensors"],"expectedSizeBytes":4,
        "expectedSha256":"770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c", "coRequisite":true,"required":"soft"});
    model["downloads"].as_array_mut().unwrap().push(row.clone());
    seed_cache(data.path(), repo, &["q4/model.safetensors"]);
    let state = || install_state_for(model_download_context(&model).unwrap(), &model, data.path());
    assert!(state().installed && state().update_available);
    let path = huggingface_repo_cache_path(data.path(), "test/predictor")
        .unwrap()
        .join("snapshots")
        .join("a".repeat(40))
        .join("predictor.safetensors");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"good").unwrap();
    assert!(
        co_requisite_cache_health(data.path(), &row)
            .unwrap()
            .installed
    );
    std::fs::write(&path, b"evil").unwrap();
    assert!(
        !co_requisite_cache_health(data.path(), &row)
            .unwrap()
            .installed
    );
    assert!(state().installed && state().update_available);
}

#[test]
fn optional_tier_dependencies_do_not_block_an_installed_primary() {
    let _env = isolate_hf_cache();
    let data = tempfile::tempdir().unwrap();
    let repo = "SceneWorks/matrix";
    let mut model = quant_matrix_model(repo);
    model["downloads"].as_array_mut().unwrap().push(json!({
        "provider": "huggingface", "repo": "SceneWorks/optional", "coRequisite": true,
        "variant": "q4", "required": "soft", "files": ["optional.bin"]
    }));
    seed_cache(data.path(), repo, &["q4/model.safetensors"]);
    let variants = model_variant_states(&model, data.path());
    let q4 = variants.iter().find(|v| v.variant == "q4").unwrap();
    assert!(q4.installed && !q4.cache_incomplete);
    let state = install_state_for(model_download_context(&model).unwrap(), &model, data.path());
    assert!(state.installed && !state.cache_incomplete);
}

#[test]
fn detects_matrix_and_single_variant_shapes() {
    // A variant-keyed multi-entry model → a matrix.
    assert!(model_has_variant_matrix(&quant_matrix_model(
        "SceneWorks/matrix"
    )));
    // A single entry with an explicit variant → still a matrix (tier-tracked).
    assert!(model_has_variant_matrix(&json!({
        "downloads": [{ "provider": "huggingface", "repo": "o/m", "variant": "q4" }]
    })));
    // A single unlabeled entry → NOT a matrix (back-compat single-variant).
    assert!(!model_has_variant_matrix(&json!({
        "downloads": [{ "provider": "huggingface", "repo": "o/m" }]
    })));
    // MULTIPLE unlabeled entries (alternate sources / co-requisite TE repos / native fallback)
    // → NOT a matrix. Entry count is not the discriminator; only an explicit `variant` is
    // (sc-8508). Guards against the old `supported.len() > 1` heuristic that falsely flagged
    // ~30 multi-repo models.
    assert!(!model_has_variant_matrix(&json!({
        "downloads": [
            { "provider": "huggingface", "repo": "org/backbone" },
            { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it" }
        ]
    })));
    // An empty-string variant is not a real tier label → not a matrix.
    assert!(!model_has_variant_matrix(&json!({
        "downloads": [
            { "provider": "huggingface", "repo": "org/a", "variant": "" },
            { "provider": "huggingface", "repo": "org/b" }
        ]
    })));
    // No downloads → not a matrix.
    assert!(!model_has_variant_matrix(&json!({ "id": "x" })));
}

#[test]
fn alternate_source_multi_entry_yields_one_default_variant() {
    // Two unlabeled download entries (alternate sources / co-requisite TE repo) must NOT be
    // treated as a quant matrix: both would otherwise collapse to a duplicate "default" key.
    // The dedup guard emits exactly one "default" variant, matching the single-variant contract.
    let _env = isolate_hf_cache(); // resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let model = json!({
        "id": "alt_source",
        "downloads": [
            { "provider": "huggingface", "repo": "org/backbone" },
            { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it" }
        ]
    });
    assert!(!model_has_variant_matrix(&model));
    let states = model_variant_states(&model, data_dir);
    assert_eq!(states.len(), 1, "alternate-source model emits one variant");
    assert_eq!(states[0].variant, "default");
}

#[test]
fn variant_keys_are_unique_across_emitted_states() {
    // Every emitted variant key must be unique. A manifest that duplicates a variant (or maps
    // two entries to the same key) keeps only the first; downstream tracking never emits two
    // same-keyed states (sc-8508).
    let _env = isolate_hf_cache(); // resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();

    // Genuine matrix → three distinct keys.
    let matrix = quant_matrix_model("SceneWorks/matrix");
    let states = model_variant_states(&matrix, data_dir);
    let mut keys: Vec<_> = states.iter().map(|s| s.variant.clone()).collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys, vec!["bf16", "q4", "q8"]);

    // Two entries sharing a variant key → collapsed to one (first wins).
    let dup = json!({
        "id": "dup",
        "downloads": [
            { "provider": "huggingface", "repo": "org/a", "variant": "q4", "files": ["q4/*"] },
            { "provider": "huggingface", "repo": "org/b", "variant": "q4", "files": ["q4-alt/*"] }
        ]
    });
    let dup_states = model_variant_states(&dup, data_dir);
    assert_eq!(dup_states.len(), 1);
    assert_eq!(dup_states[0].variant, "q4");
}

#[test]
fn single_variant_model_yields_one_default_variant() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let model = json!({
        "id": "single",
        "downloads": [{ "provider": "huggingface", "repo": "owner/single" }]
    });
    // Not installed yet.
    let states = model_variant_states(&model, data_dir);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0].variant, "default");
    assert!(!states[0].installed);

    // Seed the cache with a payload file → installed.
    seed_cache(data_dir, "owner/single", &["model.safetensors"]);
    let states = model_variant_states(&model, data_dir);
    assert!(states[0].installed);
    assert!(states[0].installed_path.is_some());
}

fn supra2_convert_entry() -> serde_json::Map<String, Value> {
    json!({
        "id": "supra2_img",
        "mlx": {
            "requiresConversion": true,
            "converter": "supra2_img",
            "convertSourceRepo": "SupraLabs/Supra2-IMG",
            "convertSourceFile": "model_final_ema.pt"
        }
    })
    .as_object()
    .unwrap()
    .clone()
}

/// sc-24216: a Supra2 dir that carries the install marker but fails the engine's pin contract
/// (here: no bundle manifest at all — what an interrupted or pre-contract convert leaves) reads
/// as NEEDING CONVERSION with the engine's reason, never as installed, so the Model Manager
/// offers Convert again and no `modelPath` is injected for a bundle that cannot load. A
/// converter with no declared contract keeps the marker-only verdict.
#[test]
fn a_marked_but_broken_supra2_bundle_reads_as_needing_conversion_with_the_reason() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let model = supra2_convert_entry();
    seed_cache(data_dir, "SupraLabs/Supra2-IMG", &["model_final_ema.pt"]);
    let dir = data_dir.join("models").join("mlx").join("supra2_img");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model_index.json"), b"{}").unwrap();

    let status = mlx_catalog_status(&model, data_dir).expect("status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "needs_conversion");
    assert!(
        status.converted_path.is_none(),
        "no modelPath for a broken bundle"
    );
    let problem = status
        .conversion_problem
        .expect("the engine's refusal reason");
    assert!(
        problem.contains("is not a converted Supra2 bundle"),
        "{problem}"
    );

    // Same marker-only shape for a converter WITHOUT a contract: unchanged, installed.
    let mut other = model.clone();
    other.insert("id".to_owned(), json!("other_convert"));
    other["mlx"]["converter"] = json!("flux2_klein_diffusers");
    seed_converted(data_dir, "other_convert");
    let status = mlx_catalog_status(&other, data_dir).expect("status");
    assert_eq!(status.install_state, "installed");
    assert!(status.conversion_problem.is_none());
}

/// sc-24216: a whole-model delete of a convert-at-install entry reclaims its converted dir along
/// with the source checkpoint, but never the shared co-requisite components (Supra2's
/// Flan-T5-Base and SD VAE may serve other models).
#[test]
fn whole_model_delete_paths_cover_the_converted_dir_but_not_shared_components() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let model = json!({
        "id": "supra2_img",
        "mlx": { "requiresConversion": true, "converter": "supra2_img" },
        "downloads": [
            { "provider": "huggingface", "repo": "SupraLabs/Supra2-IMG", "default": true,
              "files": ["model_final_ema.pt"] },
            { "provider": "huggingface", "repo": "google/flan-t5-base", "coRequisite": true,
              "componentId": "text_encoder", "files": ["config.json"] },
            { "provider": "huggingface", "repo": "stabilityai/sd-vae-ft-mse",
              "coRequisite": true, "componentId": "vae", "files": ["config.json"] }
        ]
    });
    let paths = model_artifact_paths(&model, data_dir);
    assert!(paths.contains(&data_dir.join("models").join("mlx").join("supra2_img")));
    assert!(paths.contains(&huggingface_repo_cache_path(data_dir, "SupraLabs/Supra2-IMG").unwrap()));
    for shared in ["google/flan-t5-base", "stabilityai/sd-vae-ft-mse"] {
        let cache = huggingface_repo_cache_path(data_dir, shared).unwrap();
        assert!(
            !paths.contains(&cache),
            "{shared} is shared and must survive"
        );
        assert!(!paths.contains(&data_dir.join("models").join(safe_download_dir(shared))));
    }
    // A turnkey entry owns no converted dir; an id that is not one path component is never
    // joined under the conversion root.
    let turnkey = json!({ "id": "turnkey", "mlx": { "repo": "org/turnkey" } });
    assert!(!model_artifact_paths(&turnkey, data_dir)
        .iter()
        .any(|path| path.starts_with(data_dir.join("models").join("mlx"))));
    let escaping = json!({ "id": "../escape", "mlx": { "requiresConversion": true } });
    assert_eq!(converted_artifact_dir(&escaping, data_dir), None);
}

/// sc-24216 review: the converted-dir reclaim applies to EVERY `requiresConversion` entry, not
/// only Supra2. An Anima-shaped converted root (marker + one subdir per quant tier) is removed
/// whole and the catalog drops back to not-installed.
#[tokio::test]
async fn whole_model_delete_reclaims_an_anima_shaped_converted_root() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    // An Anima-shaped convert-at-install entry: the converted root holds the marker plus one
    // subdir per quant tier, and the source repo is the shared `circlestone-labs/Anima`.
    let model = json!({
        "id": "anima_base",
        "mlx": {
            "requiresConversion": true,
            "converter": "anima_quant",
            "convertSourceRepo": "circlestone-labs/Anima"
        },
        "downloads": [
            { "provider": "huggingface", "repo": "circlestone-labs/Anima", "default": true,
              "files": ["split_files/diffusion_models/anima-base.safetensors"] }
        ]
    });
    seed_cache(
        data_dir,
        "circlestone-labs/Anima",
        &["split_files/diffusion_models/anima-base.safetensors"],
    );
    let converted = data_dir.join("models").join("mlx").join("anima_base");
    for tier in ["bf16", "q8", "q4"] {
        let dm = converted.join(tier).join("diffusion_models");
        std::fs::create_dir_all(&dm).unwrap();
        std::fs::write(dm.join("dit.safetensors"), b"weights").unwrap();
    }
    std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
    let status = mlx_catalog_status(model.as_object().unwrap(), data_dir).expect("status");
    assert_eq!(status.install_state, "installed");

    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
    remove_whole_model_artifacts(
        std::slice::from_ref(&model),
        "anima_base",
        &model,
        data_dir,
        &allowed_roots,
        true,
    )
    .await
    .expect("delete succeeds");
    assert!(
        !converted.exists(),
        "every tier and the marker are reclaimed"
    );
    let status = mlx_catalog_status(model.as_object().unwrap(), data_dir).expect("status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "needs_source");
}

/// sc-24216 review: a SAME-SIZE corruption passes the catalog's cheap check, so the card read
/// "Converted" (Convert disabled) while every job was refused. Once a failed load's full
/// verification records the fault (what the worker's `explain_failed_load` does), the catalog
/// reports `needs_conversion` + `mlxConversionProblem`, which enables "Convert again".
#[test]
fn a_recorded_digest_failure_flips_a_same_size_corrupt_bundle_to_needs_conversion() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let model = supra2_convert_entry();
    seed_cache(data_dir, "SupraLabs/Supra2-IMG", &["model_final_ema.pt"]);
    let bundle = data_dir.join("models").join("mlx").join("supra2_img");
    sceneworks_core::converted_artifacts::write_same_size_corrupt_supra2_bundle(&bundle).unwrap();
    let status = mlx_catalog_status(&model, data_dir).expect("status");
    assert_eq!(
        status.conversion_state, "converted",
        "sizes alone cannot see it — the gap the record closes"
    );

    let problem = sceneworks_core::converted_artifacts::converted_artifact_problem(
        "supra2_img",
        &bundle,
        sceneworks_core::converted_artifacts::ArtifactCheck::Digests,
    )
    .expect("the digests catch the corruption");
    sceneworks_core::converted_artifacts::record_verification_failure(&bundle, &problem).unwrap();
    let status = mlx_catalog_status(&model, data_dir).expect("status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "needs_conversion");
    assert!(status.converted_path.is_none());
    assert_eq!(status.conversion_problem.as_deref(), Some(problem.as_str()));
}

/// sc-24216, end to end through the whole-model delete's removal: the converted bundle and the
/// source checkpoint go, the installed co-requisites (shareable with other models) stay, and
/// the catalog no longer reads the model as converted.
#[tokio::test]
async fn whole_model_delete_reclaims_the_converted_bundle_and_keeps_shared_components() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let model = json!({
        "id": "supra2_img",
        "mlx": {
            "requiresConversion": true,
            "converter": "supra2_img",
            "convertSourceRepo": "SupraLabs/Supra2-IMG",
            "convertSourceFile": "model_final_ema.pt"
        },
        "downloads": [
            { "provider": "huggingface", "repo": "SupraLabs/Supra2-IMG", "default": true,
              "files": ["model_final_ema.pt"] },
            { "provider": "huggingface", "repo": "google/flan-t5-base", "coRequisite": true,
              "componentId": "text_encoder", "files": ["config.json"] },
            { "provider": "huggingface", "repo": "stabilityai/sd-vae-ft-mse",
              "coRequisite": true, "componentId": "vae", "files": ["config.json"] }
        ]
    });
    seed_cache(data_dir, "SupraLabs/Supra2-IMG", &["model_final_ema.pt"]);
    seed_cache(data_dir, "google/flan-t5-base", &["config.json"]);
    seed_cache(data_dir, "stabilityai/sd-vae-ft-mse", &["config.json"]);
    let converted = data_dir.join("models").join("mlx").join("supra2_img");
    std::fs::create_dir_all(&converted).unwrap();
    std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
    std::fs::write(converted.join("denoiser.safetensors"), b"weights").unwrap();

    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
    let removal = remove_whole_model_artifacts(
        std::slice::from_ref(&model),
        "supra2_img",
        &model,
        data_dir,
        &allowed_roots,
        true,
    )
    .await
    .expect("delete succeeds");
    assert!(!removal.removed_paths.is_empty());
    assert!(!converted.exists(), "the converted bundle is reclaimed");
    assert!(
        !huggingface_repo_cache_path(data_dir, "SupraLabs/Supra2-IMG")
            .unwrap()
            .exists()
    );
    for shared in ["google/flan-t5-base", "stabilityai/sd-vae-ft-mse"] {
        assert!(
            huggingface_repo_cache_path(data_dir, shared)
                .unwrap()
                .exists(),
            "{shared} survives the delete"
        );
    }
    let status = mlx_catalog_status(model.as_object().unwrap(), data_dir).expect("status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "needs_source");
}

/// Mark a convert-at-install model "converted" by writing its local MLX `config.json`.
fn seed_converted(data_dir: &FsPath, model_id: &str) {
    let dir = data_dir.join("models").join("mlx").join(model_id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
}

/// A converted convert-at-install model reports `updateAvailable` iff the manifest's current
/// `convertSourceFile` is NOT in the source cache (the converted dir carries no version stamp,
/// so the cache is the proxy). Generic: keys only off the manifest fields.
#[test]
fn mlx_update_available_tracks_source_file_in_cache() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "TenStrip/LTX2.3-10Eros";
    let model = json!({
        "id": "ltx_2_3_eros",
        "mlx": {
            "requiresConversion": true,
            "converter": "ltx_video",
            "convertSourceRepo": repo,
            "convertSourceFile": "10Eros_v1.3_bf16.safetensors"
        }
    })
    .as_object()
    .unwrap()
    .clone();

    // Not converted + nothing cached → not installed, no update signal.
    let status = mlx_catalog_status(&model, data_dir).expect("status");
    assert_eq!(status.install_state, "missing");
    assert!(!status.update_available);

    // Converted, but only the OLDER source is cached (manifest now points at v1.3) → stale.
    seed_converted(data_dir, "ltx_2_3_eros");
    seed_cache(data_dir, repo, &["10Eros_v1_bf16.safetensors"]);
    let status = mlx_catalog_status(&model, data_dir).expect("status");
    assert_eq!(status.install_state, "installed");
    assert_eq!(status.conversion_state, "converted");
    assert!(
        status.update_available,
        "current source not cached → update available"
    );

    // The manifest's current source file is now cached → up to date.
    seed_cache(data_dir, repo, &["10Eros_v1.3_bf16.safetensors"]);
    let status = mlx_catalog_status(&model, data_dir).expect("status");
    assert!(
        !status.update_available,
        "current source cached → no update"
    );
}

/// A dir-based converter (no `convertSourceFile`) never reports an update — the mechanism
/// degrades to a no-op rather than misfiring, so it's safe to leave enabled for all models.
#[test]
fn mlx_update_unavailable_without_convert_source_file() {
    let _env = isolate_hf_cache(); // resolve the convert-source cache under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let model = json!({
        "id": "flux2_dev",
        "mlx": {
            "requiresConversion": true,
            "converter": "flux2_dev_quant",
            "convertSourceRepo": "black-forest-labs/FLUX.2-dev"
        }
    })
    .as_object()
    .unwrap()
    .clone();
    seed_converted(data_dir, "flux2_dev");
    let status = mlx_catalog_status(&model, data_dir).expect("status");
    assert_eq!(status.conversion_state, "converted");
    assert!(
        !status.update_available,
        "no convertSourceFile → never reports an update"
    );
}

/// sc-20529: which convert-at-install entries get MLX status surfaced. The converter half is
/// delegated to `sceneworks_core::jobs_store::convert_artifact_required_here` — the same
/// predicate the worker's preflight gates on — so this pins the DELEGATION (identical verdicts
/// on both sides of the API/worker seam) as well as the entry-shape half that stays local.
#[test]
fn converted_artifact_predicate_tracks_the_core_platform_rule() {
    use sceneworks_core::jobs_store::convert_artifact_required_here;
    let entry = |value: Value| value.as_object().unwrap().clone();

    // flux2_klein_9b_true_v2: the candle converter exists (sc-7459), so the converted artifact
    // is required on EVERY platform — this is the entry the whole story is about.
    assert!(entry_requires_converted_artifact_here(&entry(json!({
        "id": "flux2_klein_9b_true_v2",
        "mlx": { "requiresConversion": true, "converter": "flux2_klein_diffusers" }
    }))));

    // Anima / LTX / the prequant converters are macOS-only, so off-Mac they require NO converted
    // artifact: surfacing them there would report `missing` for models that legitimately load
    // their raw source tree. On macOS every native converter does produce the artifact. Asserted
    // against the core helper rather than a second hard-coded membership test — that duplication
    // is exactly what this delegation removed.
    for converter in [
        "anima_quant",
        "ltx_video",
        "flux2_dev_quant",
        "sd3_5_large_quant",
    ] {
        assert_eq!(
            entry_requires_converted_artifact_here(&entry(json!({
                "id": "x",
                "mlx": { "requiresConversion": true, "converter": converter }
            }))),
            convert_artifact_required_here(converter),
            "{converter}: the catalog predicate must agree with the core platform rule"
        );
    }
    // …and off macOS that verdict is concretely `false` (the Anima trap, stated outright).
    #[cfg(not(target_os = "macos"))]
    assert!(!entry_requires_converted_artifact_here(&entry(json!({
        "id": "anima_base",
        "mlx": { "requiresConversion": true, "converter": "anima_quant" }
    }))));

    // The entry-shape half, which stays local: a turnkey that merely names a converter is not
    // convert-at-install, even on macOS where the converter itself qualifies.
    assert!(!entry_requires_converted_artifact_here(&entry(json!({
        "id": "x",
        "mlx": { "converter": "flux2_klein_diffusers" }
    }))));

    // No mlx block at all.
    assert!(!entry_requires_converted_artifact_here(&entry(
        json!({ "id": "x" })
    )));
}

/// The off-Mac surfacing end-to-end (sc-20529): on EVERY platform, an unconverted
/// `flux2_klein_9b_true_v2` reports `needs_source`/`needs_conversion` and a converted one
/// reports `converted` + its path, so the Studio can offer the convert affordance and never
/// present the model as ready off its raw single-file download.
#[test]
fn convert_at_install_klein_reports_mlx_state_on_every_platform() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let mut model = json!({
        "id": "flux2_klein_9b_true_v2",
        "name": "FLUX.2 [klein] 9B True V2",
        "mlx": {
            "requiresConversion": true,
            "converter": "flux2_klein_diffusers",
            "convertSourceRepo": "wikeeyang/Flux2-Klein-9B-True-V2",
            "convertSourceFile": "Flux2-Klein-9B-True-v2-bf16.safetensors"
        }
    })
    .as_object()
    .unwrap()
    .clone();

    apply_mac_and_mlx_fields(&mut model, data_dir);
    assert_eq!(
        model.get("mlxConversionState").and_then(Value::as_str),
        Some("needs_source"),
        "an unconverted klein must report its conversion state on every platform"
    );
    assert_eq!(
        model.get("mlxInstallState").and_then(Value::as_str),
        Some("missing")
    );

    seed_converted(data_dir, "flux2_klein_9b_true_v2");
    let mut model = model.clone();
    model.remove("mlxConversionState");
    apply_mac_and_mlx_fields(&mut model, data_dir);
    assert_eq!(
        model.get("mlxConversionState").and_then(Value::as_str),
        Some("converted")
    );
    assert_eq!(
        model.get("mlxInstallState").and_then(Value::as_str),
        Some("installed")
    );
    assert!(model
        .get("mlxConvertedPath")
        .and_then(Value::as_str)
        .is_some_and(|path| path.ends_with("flux2_klein_9b_true_v2")));
}

/// sc-24214: Supra2-IMG's converter is backend-neutral and `candle-gen-supra2` loads the bundle it
/// writes, so the shipped entry reports its conversion state on EVERY platform: `needs_conversion`
/// with the pinned `.pt` cached but unconverted, `converted` + the bundle path once the convert ran
/// — the Windows/Linux Model Manager offers the same convert step macOS does.
///
/// sc-24216: "once the convert ran" means a bundle that satisfies the engine's pin contract, not
/// a bare install marker. A marker-only dir (an interrupted or pre-contract convert) therefore
/// stays `needs_conversion` with the engine's reason on EVERY platform; the positive `converted`
/// path needs the real pinned conversion and is proven by the nax-macos-2 real-weight profile
/// (`supra2-mlx-real-weights`).
#[test]
fn supra2_reports_its_conversion_state_on_every_platform() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let manifest = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
        .expect("valid embedded model catalog");
    let shipped = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == "supra2_img")
        .and_then(Value::as_object)
        .cloned()
        .expect("supra2_img is in the builtin catalog");
    assert!(entry_requires_converted_artifact_here(&shipped));

    seed_cache(data_dir, "SupraLabs/Supra2-IMG", &["model_final_ema.pt"]);
    let mut model = shipped.clone();
    apply_mac_and_mlx_fields(&mut model, data_dir);
    assert_eq!(
        model.get("mlxConversionState").and_then(Value::as_str),
        Some("needs_conversion"),
        "a downloaded but unconverted Supra2 checkpoint must say so on every platform"
    );
    assert_eq!(
        model.get("mlxInstallState").and_then(Value::as_str),
        Some("missing")
    );

    seed_converted(data_dir, "supra2_img");
    let mut model = shipped;
    apply_mac_and_mlx_fields(&mut model, data_dir);
    assert_eq!(
        model.get("mlxConversionState").and_then(Value::as_str),
        Some("needs_conversion"),
        "a marker without a pin-contract bundle must not read as converted"
    );
    assert!(model
        .get("mlxConversionProblem")
        .and_then(Value::as_str)
        .is_some_and(|problem| problem.contains("is not a converted Supra2 bundle")));
    assert_eq!(model.get("mlxConvertedPath"), Some(&Value::Null));
}

/// The Anima non-regression (sc-20529). Off macOS, Anima must keep reporting NO mlx status:
/// `anima_quant` is macOS-only, so off-Mac Anima loads the raw `circlestone-labs/Anima`
/// `split_files/` tree with no converted dir. Emitting `mlxInstallState: "missing"` there
/// would present three installed, working models as not installed.
#[cfg(not(target_os = "macos"))]
#[test]
fn off_mac_anima_reports_no_mlx_conversion_state() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let mut model = json!({
        "id": "anima_base",
        "mlx": {
            "requiresConversion": true,
            "converter": "anima_quant",
            "convertSourceRepo": "circlestone-labs/Anima"
        }
    })
    .as_object()
    .unwrap()
    .clone();

    apply_mac_and_mlx_fields(&mut model, data_dir);
    assert!(
        model.get("mlxInstallState").is_none(),
        "off-Mac Anima has no converted artifact by design — reporting an MLX install state \
         would flip three working models to missing"
    );
    assert!(model.get("mlxConversionState").is_none());
}

#[test]
fn per_variant_tracking_reports_which_tiers_are_on_disk() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/matrix";
    let model = quant_matrix_model(repo);

    // Only bf16 is downloaded (its `files` filter matches only the bf16/ tree).
    seed_cache(data_dir, repo, &["bf16/model.safetensors"]);

    let states = model_variant_states(&model, data_dir);
    assert_eq!(states.len(), 3);
    let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

    // bf16 present; q4 + q8 absent — the whole point of per-variant tracking.
    assert!(by_variant("bf16").installed, "bf16 should read installed");
    assert!(!by_variant("q4").installed, "q4 should read missing");
    assert!(!by_variant("q8").installed, "q8 should read missing");

    // Footprint + size flow through: q4 uses footprint.diskSizeBytes, bf16 uses estimatedSizeBytes.
    assert_eq!(by_variant("q4").download_size_bytes, Some(5_000_000_000));
    assert_eq!(by_variant("bf16").download_size_bytes, Some(20_000_000_000));
    assert_eq!(
        by_variant("q8").footprint.get("peakMemoryBytes"),
        Some(&Value::Null)
    );
}

/// Seed one quant tier as a diffusers pipeline snapshot: always a `model_index.json` +
/// weightless scheduler/tokenizer configs; the transformer/vae/text_encoder weights only when
/// `complete`. A `complete: false` tier mirrors a torn download (interrupted, or weights pruned)
/// — its files satisfy the coarse `<tier>/*` glob but it cannot load.
fn seed_diffusers_tier(data_dir: &FsPath, repo: &str, tier: &str, complete: bool) {
    seed_diffusers_tier_rev(data_dir, repo, "abc123", tier, complete);
}

fn seed_diffusers_tier_rev(data_dir: &FsPath, repo: &str, rev: &str, tier: &str, complete: bool) {
    let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
    let root = cache.join("snapshots").join(rev).join(tier);
    let write = |rel: &str, body: &str| {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    };
    write(
        "model_index.json",
        r#"{
            "_class_name": "ZImagePipeline",
            "transformer": ["diffusers", "ZImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "Qwen3Model"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"]
        }"#,
    );
    // Weightless components ship only config, present in a torn tier too (this is what makes the
    // coarse glob match and wrongly report "installed").
    write("scheduler/scheduler_config.json", "{}");
    write("tokenizer/tokenizer_config.json", "{}");
    write("text_encoder/config.json", "{}");
    if complete {
        write("transformer/config.json", "{}");
        write("transformer/model.safetensors", "weights");
        write("vae/config.json", "{}");
        write("vae/model.safetensors", "weights");
        write("text_encoder/model.safetensors", "weights");
    }
}

/// The regression this fix closes: a torn diffusers tier (its `model_index.json` + a config or two
/// present, but the transformer/vae weights missing) satisfied the coarse `<tier>/*` glob and so
/// reported a green "Installed" badge — then failed to load at generation with `No such file or
/// directory`. A tier must read installed only when its weight-bearing components actually hold
/// weights; an absent tier stays a clean "missing", not a repairable "incomplete".
#[test]
fn torn_diffusers_tier_reads_incomplete_not_installed() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/matrix";
    let model = quant_matrix_model(repo);

    // q4 complete (loads), q8 torn (metadata only — the transformer weights never arrived),
    // bf16 never fetched.
    seed_diffusers_tier(data_dir, repo, "q4", true);
    seed_diffusers_tier(data_dir, repo, "q8", false);

    let states = model_variant_states(&model, data_dir);
    let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

    assert!(
        by_variant("q4").installed,
        "complete q4 tier must read installed"
    );
    assert!(
        !by_variant("q8").installed,
        "torn q8 tier must NOT read installed just because its metadata files match the glob"
    );
    assert!(
        by_variant("q8").cache_incomplete,
        "a torn (half-present) tier is a repairable incomplete, not a clean missing"
    );
    assert!(
        !by_variant("bf16").installed && !by_variant("bf16").cache_incomplete,
        "a never-fetched tier stays a clean missing (no spurious repair prompt — sc-9907)"
    );

    // Model-level state aggregates: q4 is complete, so the model is INSTALLED (usable) overall —
    // and because q8 is genuinely TORN, it is also repairable (sc-14431). The complete q4 keeps
    // `installed` true (the model still works via q4); the torn q8 raises `cache_incomplete` so the
    // model-level Fix button appears and knows what to re-fetch. This is the fix to the old
    // suppression that hid a torn sibling behind a complete one — distinct from sc-9907, where the
    // OTHER tiers were merely never downloaded (missing), not torn.
    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(state.installed, "model installed via the complete q4 tier");
    assert!(
        state.cache_incomplete,
        "a genuinely torn q8 makes the model repairable even though q4 is complete (sc-14431)"
    );
    assert!(
        state
            .missing_required_files
            .iter()
            .any(|file| file.starts_with("q8/")),
        "the model-level repair must name the torn q8 tier's missing files, got {:?}",
        state.missing_required_files
    );
}

/// sc-14431: the model-level repair signal keys on a genuinely TORN tier alone, not on
/// `!any_installed`. A complete sibling must NOT suppress repair for a half-downloaded tier — but a
/// never-fetched (missing) sibling must still NOT raise a spurious repair (sc-9907 preserved).
#[test]
fn torn_tier_stays_repairable_behind_a_complete_sibling_but_missing_does_not() {
    let _env = isolate_hf_cache();
    let repo = "SceneWorks/matrix";
    let model = quant_matrix_model(repo);

    // Case A — complete q4 + TORN q8 (metadata only) + missing bf16 → repairable, naming q8.
    {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        seed_diffusers_tier(data_dir, repo, "q4", true);
        seed_diffusers_tier(data_dir, repo, "q8", false);
        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(state.installed, "usable via the complete q4 tier");
        assert!(
            state.cache_incomplete,
            "a torn q8 behind a complete q4 must be repairable (sc-14431)"
        );
        assert!(state
            .missing_required_files
            .iter()
            .any(|file| file.starts_with("q8/")));
    }

    // Case B — complete q4 + q8/bf16 NEVER fetched (missing, not torn) → installed, NOT repairable.
    // This is the sc-9907 scenario the fix must not regress.
    {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        seed_diffusers_tier(data_dir, repo, "q4", true);
        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(state.installed, "usable via the complete q4 tier");
        assert!(
            !state.cache_incomplete,
            "a clean single-tier install with the others merely un-downloaded must NOT read \
             repairable (sc-9907)"
        );
        assert!(state.missing_required_files.is_empty());
    }
}

/// GH #1858: a manifest revision bump re-hosts a tier to add a previously-absent component (the
/// FLUX.2 MLX re-hosts' `scheduler/`, whose `model_index.json` had always declared it). The freshly
/// re-downloaded COMPLETE snapshot then coexists with the older torn snapshot it replaced — a
/// quant-matrix repo keeps every revision in ONE shared cache, and repair just re-runs the download
/// (nothing evicts the old snapshot). `huggingface_snapshot_dirs` can rank the OLD snapshot first
/// (fronted by `refs/main`, or by a higher file count), so a per-tier check that reads only the
/// first-ranked snapshot kept reporting the tier incomplete even after re-downloading. The tier
/// completeness must instead ask "is this tier complete in ANY cached revision?".
#[test]
fn a_complete_revision_rescues_a_torn_older_snapshot_of_the_same_tier() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/matrix";
    let model = quant_matrix_model(repo);

    // Old revision: every weight present but NO `scheduler/` folder — exactly the pre-#1858 FLUX.2
    // re-host (its `model_index.json` declares `scheduler`, so it reads torn/incomplete).
    seed_diffusers_tier_rev(data_dir, repo, "oldrev", "bf16", true);
    let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
    std::fs::remove_dir_all(
        cache
            .join("snapshots")
            .join("oldrev")
            .join("bf16")
            .join("scheduler"),
    )
    .unwrap();
    // New revision: the re-host with the scheduler config added — a complete tier.
    seed_diffusers_tier_rev(data_dir, repo, "newrev", "bf16", true);
    // Reproduce the worst-case ranking: front the OLD (torn) snapshot via refs/main.
    std::fs::create_dir_all(cache.join("refs")).unwrap();
    std::fs::write(cache.join("refs").join("main"), "oldrev").unwrap();

    let states = model_variant_states(&model, data_dir);
    let bf16 = states.iter().find(|s| s.variant == "bf16").unwrap();
    assert!(
        bf16.installed && !bf16.cache_incomplete,
        "a complete cached revision must make the tier read installed even when a torn older \
         snapshot is fronted by refs/main — got installed={} incomplete={} missing={:?}",
        bf16.installed,
        bf16.cache_incomplete,
        bf16.missing_required_files
    );
}

/// A SANA quant-matrix turnkey (family `sana`). Ships NO `model_index.json`, so the diffusers
/// completeness augmentation is a no-op — the tightening comes from the shared `sana_tier_complete`.
fn sana_matrix_model(repo: &str) -> Value {
    json!({
        "id": "sana_sprint_1600m",
        "family": "sana",
        "downloads": [
            { "provider": "huggingface", "repo": repo, "variant": "q4", "default": true, "files": ["q4/*"] },
            { "provider": "huggingface", "repo": repo, "variant": "q8", "files": ["q8/*"] },
            { "provider": "huggingface", "repo": repo, "variant": "bf16", "files": ["bf16/*"] }
        ]
    })
}

/// Seed a SANA quant tier: always the transformer + VAE weights, plus the Gemma text encoder + its
/// tokenizer only when `complete`. A `complete: false` tier is TORN — its `<tier>/*` glob matches
/// (transformer present) but the loader dies on the absent text encoder / tokenizer.
fn seed_sana_tier(data_dir: &FsPath, repo: &str, tier: &str, complete: bool) {
    let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
    let root = cache.join("snapshots").join("abc123").join(tier);
    let write = |rel: &str| {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    };
    write("transformer/diffusion_pytorch_model.safetensors");
    write("vae/diffusion_pytorch_model.safetensors");
    if complete {
        write("text_encoder/gemma-2-2b-it.safetensors");
        write("text_encoder/tokenizer.json");
    }
}

/// The sc-13513 fix for a no-`model_index` quant-matrix turnkey: a torn SANA tier (its `<tier>/*`
/// glob matches the transformer, but the Gemma text encoder + tokenizer never landed) reported a
/// green "Installed" badge, then failed to load. It must read `incomplete`; a never-fetched tier
/// stays a clean `missing`.
#[test]
fn torn_sana_tier_reads_incomplete_not_installed() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/Sana_Sprint_1.6B_1024px_mlx";
    let model = sana_matrix_model(repo);

    // q4 complete (loads), q8 torn (text encoder + tokenizer missing), bf16 never fetched.
    seed_sana_tier(data_dir, repo, "q4", true);
    seed_sana_tier(data_dir, repo, "q8", false);

    let states = model_variant_states(&model, data_dir);
    let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

    assert!(
        by_variant("q4").installed,
        "complete q4 tier must read installed"
    );
    assert!(
        !by_variant("q8").installed,
        "torn q8 tier must NOT read installed just because its transformer matched the glob"
    );
    assert!(
        by_variant("q8").cache_incomplete,
        "a torn (half-present) tier is a repairable incomplete, not a clean missing"
    );
    assert!(
        !by_variant("bf16").installed && !by_variant("bf16").cache_incomplete,
        "a never-fetched tier stays a clean missing (no spurious repair prompt)"
    );

    // Mutation check: completing q8 (add the Gemma text encoder + tokenizer) flips it to installed,
    // proving the predicate discriminates on the text encoder, not merely the transformer glob.
    seed_sana_tier(data_dir, repo, "q8", true);
    let states = model_variant_states(&model, data_dir);
    assert!(states.iter().find(|s| s.variant == "q8").unwrap().installed);
}

/// A SenseNova-U1 quant-matrix turnkey (family `sensenova-u1`). Ships a FLAT unified tier — packed
/// backbone + `config.json` + tokenizer files at the tier root, NO `model_index.json` and no
/// component subdirs — so the diffusers augmentation is a no-op and the tightening comes from the
/// shared `sensenova_tier_complete` (sc-14432).
fn sensenova_matrix_model(repo: &str) -> Value {
    json!({
        "id": "sensenova_u1_8b",
        "family": "sensenova-u1",
        "downloads": [
            { "provider": "huggingface", "repo": repo, "variant": "q4", "default": true, "files": ["q4/*"] },
            { "provider": "huggingface", "repo": repo, "variant": "q8", "files": ["q8/*"] },
            { "provider": "huggingface", "repo": repo, "variant": "bf16", "files": ["bf16/*"] }
        ]
    })
}

/// Seed a SenseNova quant tier: always the packed backbone + `config.json`, plus its own
/// `tokenizer.json` only when `with_tokenizer`. A tokenizer-less tier still matches its `<tier>/*`
/// glob on the backbone, and loads ONLY if a sibling tier carries a tokenizer to borrow.
fn seed_sensenova_tier(data_dir: &FsPath, repo: &str, tier: &str, with_tokenizer: bool) {
    let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
    let root = cache.join("snapshots").join("abc123").join(tier);
    let write = |rel: &str| {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    };
    write("model.safetensors");
    write("config.json");
    if with_tokenizer {
        write("tokenizer.json");
    }
}

/// sc-14432: a SenseNova tier whose `tokenizer.json` never landed cleared the coarse `<tier>/*` glob
/// on its backbone and reported a green "Installed" badge, then died at load on the absent
/// tokenizer — "complete but unloadable", with no repair offered. It must read `incomplete`, UNLESS a
/// sibling tier carries a tokenizer the engine can borrow (`resolve_tokenizer_path`), in which case
/// the tier genuinely loads and must keep reading installed.
#[test]
fn tokenizerless_sensenova_tier_reads_incomplete_unless_a_sibling_can_lend_one() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/sensenova-u1-8b-mlx";
    let model = sensenova_matrix_model(repo);

    // q4 alone, and it ships no tokenizer: nothing to borrow, so it cannot load.
    seed_sensenova_tier(data_dir, repo, "q4", false);
    let states = model_variant_states(&model, data_dir);
    let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();
    assert!(
        !by_variant("q4").installed,
        "a tokenizer-less q4 with no sibling to borrow from must NOT read installed"
    );
    assert!(
        by_variant("q4").cache_incomplete,
        "it is a repairable incomplete, not a clean missing"
    );
    assert!(
        !by_variant("bf16").installed && !by_variant("bf16").cache_incomplete,
        "a never-fetched tier stays a clean missing (no spurious repair prompt)"
    );

    // Mutation check: installing q8 (which DOES ship the tokenizer) makes q4 loadable by borrowing
    // it — proving the predicate tracks the engine's sibling resolution, not mere file presence.
    seed_sensenova_tier(data_dir, repo, "q8", true);
    let states = model_variant_states(&model, data_dir);
    let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();
    assert!(
        by_variant("q4").installed,
        "q4 borrows q8's tokenizer, so it loads and must read installed"
    );
    assert!(by_variant("q8").installed);
}

/// A Boogu turnkey (family `boogu`): a single unlabeled "default" download whose `files` filter
/// enumerates the three component subdirs of the shipped `base/` Q8 subfolder.
fn boogu_model(repo: &str) -> Value {
    json!({
        "id": "boogu_image",
        "family": "boogu",
        "downloads": [
            { "provider": "huggingface", "repo": repo, "files": ["base/transformer/*", "base/mllm/*", "base/vae/*"] }
        ]
    })
}

/// Seed a Boogu `base/` subfolder: transformer (weights + config), mllm weights, and VAE, plus the
/// `mllm/tokenizer.json` only when `complete`. A `complete: false` tree is TORN — the loader crashes
/// first on the missing tokenizer, but each `base/<dir>/*` glob matches the stray weights.
fn seed_boogu_default(data_dir: &FsPath, repo: &str, complete: bool) {
    let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
    let base = cache.join("snapshots").join("abc123").join("base");
    let write = |rel: &str| {
        let path = base.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    };
    write("transformer/diffusion_pytorch_model.safetensors");
    write("transformer/config.json");
    write("mllm/model.safetensors");
    write("vae/diffusion_pytorch_model.safetensors");
    if complete {
        write("mllm/tokenizer.json");
    }
}

#[test]
fn torn_boogu_default_reads_incomplete_not_installed() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/boogu-image-mlx";
    let model = boogu_model(repo);

    // Torn: transformer + mllm weights + vae present, but `mllm/tokenizer.json` never landed. Each
    // `base/<dir>/*` glob matches the stray weights, so the coarse check reads installed.
    seed_boogu_default(data_dir, repo, false);
    let states = model_variant_states(&model, data_dir);
    assert_eq!(states.len(), 1);
    assert!(
        !states[0].installed,
        "torn boogu default must NOT read installed just because the component globs matched"
    );
    assert!(
        states[0].cache_incomplete,
        "a torn boogu default is a repairable incomplete"
    );

    // Mutation check: adding the tokenizer completes it.
    seed_boogu_default(data_dir, repo, true);
    let states = model_variant_states(&model, data_dir);
    assert!(
        states[0].installed,
        "complete boogu default reads installed"
    );
}

/// The TOP-LEVEL badge must agree with the per-variant state. Boogu is a single-variant download, so
/// `install_state_for` takes the non-matrix else branch — which, before sc-13513, used the coarse
/// cache health and rendered a false-green `installState:"installed"`/`repairAvailable:false` over a
/// torn install. The shared predicate must downgrade it there too.
#[test]
fn torn_boogu_top_level_badge_reads_incomplete_not_installed() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/boogu-image-mlx";
    let model = boogu_model(repo);

    seed_boogu_default(data_dir, repo, false);
    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        !state.installed,
        "a torn boogu install must NOT read installed at the top level"
    );
    assert!(
        state.cache_incomplete,
        "a torn boogu install is a repairable incomplete at the top level"
    );

    // Completing it flips the top-level badge to installed.
    seed_boogu_default(data_dir, repo, true);
    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        state.installed,
        "a complete boogu install reads installed at the top level"
    );
    assert!(!state.cache_incomplete);
}

/// SANA repos keep TWO cached snapshots (a tiered one + an older flat one). A tier that is complete
/// in the tiered snapshot must read installed even though the flat snapshot has no tier subdir — the
/// check folds across snapshots with `any`, not `all` (guards the epic-13075 multi-snapshot trap).
#[test]
fn sana_tier_complete_in_one_of_two_snapshots_reads_installed() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/Sana_1600M_1024px_mlx";
    let model = sana_matrix_model(repo);

    // Tiered snapshot: a complete q4.
    seed_sana_tier(data_dir, repo, "q4", true);
    // A second, FLAT snapshot with weights at the root (no q4/ subdir) — must not drag q4 down.
    let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
    let flat = cache.join("snapshots").join("flat9999");
    for rel in [
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ] {
        let path = flat.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    let states = model_variant_states(&model, data_dir);
    let q4 = states.iter().find(|s| s.variant == "q4").unwrap();
    assert!(
        q4.installed,
        "q4 complete in the tiered snapshot reads installed despite a flat sibling snapshot"
    );
    assert!(!q4.cache_incomplete);
}

#[test]
fn load_time_quant_variants_share_one_complete_snapshot_predicate() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let repo = "SceneWorks/Mage-Flow";
    let downloads = ["q4", "q8", "bf16"].map(|variant| {
        json!({
            "provider": "huggingface",
            "repo": repo,
            "variant": variant,
            "files": [
                "model_index.json", "scheduler/*", "text_encoder/*",
                "transformer/*", "vae/*"
            ]
        })
    });
    let model = json!({
        "id": "mage_flow",
        "family": "mage-flow",
        "downloads": downloads
    });
    let single = json!({
        "id": "mage_flow_single",
        "family": "mage-flow",
        "downloads": [{
            "provider": "huggingface",
            "repo": repo,
            "files": [
                "model_index.json", "scheduler/*", "text_encoder/*",
                "transformer/*", "vae/*"
            ]
        }]
    });

    fn write_json(path: &FsPath, value: Value) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    }
    fn write_safetensors(path: &FsPath) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let header = br#"{"tensor":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0]);
        std::fs::write(path, bytes).unwrap();
    }
    fn seed_mage(data_dir: &FsPath, repo: &str) -> PathBuf {
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .unwrap()
            .join("snapshots/abc123");
        write_json(
            &snapshot.join("model_index.json"),
            json!({
                "_class_name": "MageFlowPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                "tokenizer": ["transformers", "AutoProcessor"],
                "transformer": ["mage_flow", "MageFlow"],
                "vae": ["mage_flow", "MageVAE"]
            }),
        );
        for (component, class) in [
            ("text_encoder", "Qwen3VLForConditionalGeneration"),
            ("vae", "MageVAE"),
        ] {
            write_json(
                &snapshot.join(component).join("config.json"),
                json!({"_class_name": class}),
            );
            write_safetensors(
                &snapshot
                    .join(component)
                    .join("diffusion_pytorch_model.safetensors"),
            );
        }
        write_json(
            &snapshot.join("transformer/config.json"),
            json!({
                "_class_name": "MageFlow",
                "in_channels": 128,
                "hidden_size": 3072,
                "depth": 12
            }),
        );
        write_safetensors(&snapshot.join("transformer/diffusion_pytorch_model.safetensors"));
        write_json(
            &snapshot.join("scheduler/scheduler_config.json"),
            json!({"_class_name": "FlowMatchEulerDiscreteScheduler"}),
        );
        write_json(
            &snapshot.join("text_encoder/tokenizer_config.json"),
            json!({"version": "1.0"}),
        );
        write_json(
            &snapshot.join("text_encoder/preprocessor_config.json"),
            json!({"size": 384}),
        );
        snapshot
    }
    let assert_health = |data_dir: &FsPath, installed: bool, label: &str| {
        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 3);
        assert!(
            states.iter().all(|state| {
                state.installed == installed && state.cache_incomplete == !installed
            }),
            "{label}: every logical variant must share one health result"
        );
        let top = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert_eq!(top.installed, installed, "{label}: matrix top-level");
        let single_top =
            install_state_for(model_download_context(&single).unwrap(), &single, data_dir);
        assert_eq!(single_top.installed, installed, "{label}: single-variant");
    };

    let valid_dir = tmp.path().join("valid");
    seed_mage(&valid_dir, repo);
    assert_health(&valid_dir, true, "complete Mage snapshot");

    let mutations = [
        ("model_index absent", "model_index.json", false),
        ("model_index malformed", "model_index.json", true),
        (
            "scheduler config absent",
            "scheduler/scheduler_config.json",
            false,
        ),
        (
            "scheduler config malformed",
            "scheduler/scheduler_config.json",
            true,
        ),
        (
            "tokenizer config absent",
            "text_encoder/tokenizer_config.json",
            false,
        ),
        (
            "tokenizer config malformed",
            "text_encoder/tokenizer_config.json",
            true,
        ),
        (
            "vision processor config absent",
            "text_encoder/preprocessor_config.json",
            false,
        ),
        (
            "vision processor config malformed",
            "text_encoder/preprocessor_config.json",
            true,
        ),
        (
            "text encoder config absent",
            "text_encoder/config.json",
            false,
        ),
        (
            "text encoder config malformed",
            "text_encoder/config.json",
            true,
        ),
        (
            "text encoder weights absent",
            "text_encoder/diffusion_pytorch_model.safetensors",
            false,
        ),
        (
            "text encoder weights malformed",
            "text_encoder/diffusion_pytorch_model.safetensors",
            true,
        ),
        (
            "transformer config absent",
            "transformer/config.json",
            false,
        ),
        (
            "transformer config malformed",
            "transformer/config.json",
            true,
        ),
        (
            "transformer weights absent",
            "transformer/diffusion_pytorch_model.safetensors",
            false,
        ),
        (
            "transformer weights malformed",
            "transformer/diffusion_pytorch_model.safetensors",
            true,
        ),
        ("VAE config absent", "vae/config.json", false),
        ("VAE config malformed", "vae/config.json", true),
        (
            "VAE weights absent",
            "vae/diffusion_pytorch_model.safetensors",
            false,
        ),
        (
            "VAE weights malformed",
            "vae/diffusion_pytorch_model.safetensors",
            true,
        ),
    ];
    for (index, (label, relative, malformed)) in mutations.into_iter().enumerate() {
        let data_dir = tmp.path().join(format!("mutation-{index}"));
        let snapshot = seed_mage(&data_dir, repo);
        let path = snapshot.join(relative);
        if malformed {
            std::fs::write(path, b"not valid").unwrap();
        } else {
            std::fs::remove_file(path).unwrap();
        }
        assert_health(&data_dir, false, label);
    }
}

/// Anima is convert-at-install; its variants[] "default" tracks the three exact `split_files/` SOURCE
/// files (not a tier subdir). `tier_subdir_name` resolves those to `split_files`, so the downgrade
/// RUNS — but must be a NO-OP: a complete source passes `anima_tier_complete` (the layout is
/// compatible), and a torn source is already coarse-missing (exact-path filter), so `if installed` is
/// false. This pins that no-op so a future divergence between the download filter and the predicate's
/// pinned filenames surfaces as a RED test, not a silent false-incomplete on the Anima source badge.
#[test]
fn anima_source_download_variant_stays_installed_predicate_is_noop() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "circlestone-labs/Anima";
    let model = json!({
        "id": "anima_base",
        "family": "anima",
        "downloads": [
            { "provider": "huggingface", "repo": repo, "files": [
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "split_files/vae/qwen_image_vae.safetensors"
            ]}
        ]
    });

    seed_cache(
        data_dir,
        repo,
        &[
            "split_files/diffusion_models/anima-base-v1.0.safetensors",
            "split_files/text_encoders/qwen_3_06b_base.safetensors",
            "split_files/vae/qwen_image_vae.safetensors",
        ],
    );
    let states = model_variant_states(&model, data_dir);
    assert_eq!(states.len(), 1);
    assert!(
        states[0].installed,
        "a complete anima source reads installed — the predicate must not false-incomplete it"
    );
    assert!(!states[0].cache_incomplete);
}

/// A model whose ONLY on-disk tier is torn must read NOT installed at the model level — including
/// via the "usable stale" receipt path (sc-13076 backfilled a receipt for whatever was on disk,
/// so a metadata-only tier produced a receipt whose files all exist yet cannot load).
#[test]
fn model_with_only_a_torn_tier_is_not_installed() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let repo = "SceneWorks/matrix";
    let model = quant_matrix_model(repo);

    // Only q8 present, and torn. Plus a backfilled receipt recording its (weightless) files as if
    // complete — exactly the shape found on the reporter's disk.
    seed_diffusers_tier(data_dir, repo, "q8", false);
    let managed = data_dir.join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&json!({
            "repo": repo,
            "receipts": [{
                "repo": repo, "modelId": "matrix_model", "variant": "q8",
                "manifestFiles": ["q8/*"],
                "resolvedFiles": [
                    "q8/model_index.json", "q8/scheduler/scheduler_config.json",
                    "q8/tokenizer/tokenizer_config.json", "q8/text_encoder/config.json"
                ],
                "backfilled": true
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        !state.installed,
        "a torn-only install must not read installed via the usable-stale receipt path"
    );
}

/// sc-24112 — a DECLARED-but-unpublished tier (`pendingArtifact`).
///
/// The catalog keeps ENUMERATING it, because the tier axis has to be real before the bytes
/// exist: the memory ladder, both fit gates, the tier picker and the download panel are all
/// built and tested against the tier that is coming. What it must never be is *installable* —
/// its revision is the null SHA, so a queued fetch cannot resolve and the user would see an
/// opaque download failure instead of the reason.
///
/// The flags below are what every downstream surface reads, so this is the one place the rule
/// is decided rather than re-derived per consumer.
#[test]
fn a_pending_artifact_tier_is_listed_but_never_installed() {
    let data = tempfile::tempdir().expect("temp data dir");
    let mut model = quant_matrix_model("SceneWorks/matrix");
    let downloads = model["downloads"].as_array_mut().expect("downloads");
    // Make q8 pending, exactly as the shipped `qwen_image_2_1` rows are.
    downloads[1]["pendingArtifact"] = json!(true);
    downloads[1]["revision"] =
        json!(sceneworks_core::model_artifacts::artifact_selection::PENDING_ARTIFACT_REVISION);

    let states = model_variant_states(&model, data.path());
    assert_eq!(
        states
            .iter()
            .map(|s| s.variant.as_str())
            .collect::<Vec<_>>(),
        vec!["q4", "q8", "bf16"],
        "a pending tier is still ENUMERATED — the tier axis is a catalog declaration, not a \
         statement about what is on disk"
    );
    let q8 = states
        .iter()
        .find(|state| state.variant == "q8")
        .expect("q8 is enumerated");
    assert!(q8.pending_artifact);
    assert!(
        !q8.installed && !q8.cache_incomplete,
        "a pending tier reads neither installed nor incomplete, whatever the cache probe \
         happened to find — there is nothing it could legitimately have found"
    );
    assert!(
        !q8.tier_deletable,
        "nothing was ever fetched, so there is nothing for the per-tier delete to reclaim"
    );
    // Its siblings are untouched: this is a per-row fact, not a switch that disables the matrix.
    for variant in ["q4", "bf16"] {
        let state = states
            .iter()
            .find(|state| state.variant == variant)
            .expect("sibling tier");
        assert!(!state.pending_artifact);
        assert!(
            state.tier_deletable,
            "{variant} carries a `files` scope, so its per-tier delete can reclaim it alone"
        );
    }
}

/// The other half of `tier_deletable`: a tier with NO `files` scope.
///
/// `DELETE /models/:id/variants/:variant` refuses it ("delete the whole model instead"),
/// because a whole-repo row IS the model rather than a slice of it — deleting it would wipe a
/// cache other tiers may share. Before sc-24112 every variant row carried a glob and the
/// delete handler's own comment assumed it always would; `qwen_image_2_1`'s bf16 tier is the
/// whole upstream snapshot and broke that assumption, so the UI needs this stated rather than
/// inferred — otherwise it renders a Delete button that always errors.
#[test]
fn a_whole_repo_tier_reports_that_it_cannot_be_reclaimed_alone() {
    let data = tempfile::tempdir().expect("temp data dir");
    let mut model = quant_matrix_model("SceneWorks/matrix");
    let downloads = model["downloads"].as_array_mut().expect("downloads");
    downloads[2]["files"] = json!([]);

    let states = model_variant_states(&model, data.path());
    let bf16 = states
        .iter()
        .find(|state| state.variant == "bf16")
        .expect("bf16 is enumerated");
    assert!(
        !bf16.tier_deletable,
        "a tier with no file scope cannot be deleted on its own; the API refuses it and the \
         UI must not offer it"
    );
    assert!(
        states
            .iter()
            .filter(|state| state.variant != "bf16")
            .all(|state| state.tier_deletable),
        "the scoped siblings still reclaim normally"
    );
}

/// sc-24112 — a scope-less tier that is the SOLE row on its repo IS reclaimable on its own: that
/// repo's whole snapshot is the tier. The split-repo layout (`qwen_image_2_1`'s bf16 is the
/// upstream snapshot, its q8/q4 the SceneWorks re-host) otherwise left a 30.86 GiB tier with no
/// per-tier delete at all. The shared-repo case above stays refused.
#[test]
fn a_scope_less_tier_alone_on_its_repo_is_deletable() {
    let data = tempfile::tempdir().expect("temp data dir");
    let mut model = quant_matrix_model("SceneWorks/matrix");
    let downloads = model["downloads"].as_array_mut().expect("downloads");
    downloads[2]["files"] = json!([]);
    downloads[2]["repo"] = json!("Upstream/dense");

    let states = model_variant_states(&model, data.path());
    assert!(
        states.iter().all(|state| state.tier_deletable),
        "the sole-repo bf16 and its scoped siblings all reclaim alone"
    );
    let bf16 = model_download_for_variant(&model, "bf16").expect("bf16 row");
    assert_eq!(
        tier_delete_scope(&model, &bf16, "bf16").expect("scoped"),
        TierDeleteScope {
            repo: "Upstream/dense".to_owned(),
            files: vec!["*".to_owned()],
            retained_files: Vec::new(),
        }
    );
    // …and the route's refusal still holds for a scope-less row whose repo IS shared.
    let mut shared = model.clone();
    shared["downloads"][0]["files"] = json!([]);
    let shared_q4 = model_download_for_variant(&shared, "q4").expect("q4 row");
    assert!(tier_delete_scope(&shared, &shared_q4, "q4").is_err());
}

#[test]
fn variant_footprint_disk_bytes_reads_required_field() {
    let entry = json!({ "footprint": { "diskSizeBytes": 42 } });
    assert_eq!(variant_footprint_disk_bytes(&entry), Some(42));
    assert_eq!(variant_footprint_disk_bytes(&json!({})), None);
}

#[test]
fn variant_download_selector_picks_the_right_tier() {
    let model = quant_matrix_model("SceneWorks/matrix");
    // Case-insensitive match on the declared variant.
    assert_eq!(
        model_download_for_variant(&model, "Q8")
            .and_then(|d| d.get("files").cloned())
            .and_then(|f| f.as_array().and_then(|a| a.first().cloned())),
        Some(Value::String("q8/*".to_owned()))
    );
    // Unknown tier → None (the handler turns this into a 400).
    assert!(model_download_for_variant(&model, "int8").is_none());
    // The default selector still picks the `default: true` (q4) entry — back-compat.
    assert_eq!(
        model_download(&model)
            .and_then(|d| d.get("variant").and_then(Value::as_str).map(str::to_owned)),
        Some("q4".to_owned())
    );
}
