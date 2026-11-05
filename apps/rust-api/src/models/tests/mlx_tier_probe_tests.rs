use super::*;

fn write_weight(dir: &std::path::Path, backbone: &str, file: &str) {
    let d = dir.join(backbone);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join(file), b"x").unwrap();
}

#[test]
fn convert_output_tiers_probes_diffusion_models_highest_first_ignoring_appledouble() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Anima layout: <tier>/diffusion_models/<dit>.safetensors present for bf16 + q8; q4 has only a
    // hidden `._` AppleDouble sidecar, which must NOT count as a loadable tier (SceneWorks#1333).
    write_weight(
        &root.join("bf16"),
        "diffusion_models",
        "anima-base-v1.0.safetensors",
    );
    write_weight(
        &root.join("q8"),
        "diffusion_models",
        "anima-base-v1.0.safetensors",
    );
    write_weight(
        &root.join("q4"),
        "diffusion_models",
        "._anima-base-v1.0.safetensors",
    );
    // Highest-fidelity first, q4 excluded.
    assert_eq!(mlx_convert_output_tiers(root), vec!["bf16", "q8"]);
}

#[test]
fn convert_output_tiers_handles_transformer_flat_and_empty_layouts() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_weight(&root.join("q8"), "transformer", "model.safetensors");
    // Flat layout: a sharded index sits directly in the tier dir (no backbone subdir).
    std::fs::create_dir_all(root.join("bf16")).unwrap();
    std::fs::write(
        root.join("bf16")
            .join("diffusion_pytorch_model.safetensors.index.json"),
        b"x",
    )
    .unwrap();
    assert_eq!(mlx_convert_output_tiers(root), vec!["bf16", "q8"]);
    // A flat converted dir (no tier subdirs) yields no tiers → the web renders no picker.
    let flat = tempfile::tempdir().unwrap();
    std::fs::write(flat.path().join("model_index.json"), b"{}").unwrap();
    assert!(mlx_convert_output_tiers(flat.path()).is_empty());
}

/// Seed a converted Anima tier: always the DiT (backbone), plus the dense text encoder + VAE only
/// when `complete`. A `complete: false` tier is TORN — its DiT satisfies the coarse
/// `tier_subdir_has_weights` probe, but the loader would die on the missing text-encoder/VAE.
fn seed_anima_tier(root: &std::path::Path, tier: &str, complete: bool) {
    let dir = root.join(tier);
    write_weight(&dir, "diffusion_models", "anima-base-v1.0.safetensors");
    if complete {
        write_weight(&dir, "text_encoders", "qwen_3_06b_base.safetensors");
        write_weight(&dir, "vae", "qwen_image_vae.safetensors");
    }
}

fn tier_state_map(states: &[Value]) -> std::collections::BTreeMap<String, (String, String)> {
    states
        .iter()
        .map(|state| {
            (
                state["tier"].as_str().unwrap().to_owned(),
                (
                    state["installState"].as_str().unwrap().to_owned(),
                    state["cacheState"].as_str().unwrap().to_owned(),
                ),
            )
        })
        .collect()
}

#[test]
fn convert_output_tier_states_mark_complete_torn_and_absent_tiers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // q4 complete (loads), q8 TORN (DiT present but text-encoder/VAE never landed), bf16 absent — all
    // THREE must appear so the picker can show each state (the "show all, disable unavailable" rule).
    seed_anima_tier(root, "q4", true);
    seed_anima_tier(root, "q8", false);

    // family "anima" gates on the shared per-tier predicate: the torn q8 reads incomplete, NOT
    // installed (sc-13513).
    let anima = mlx_convert_output_tier_states(root, "anima", "anima_base");
    assert_eq!(anima.len(), 3);
    let a = tier_state_map(&anima);
    assert_eq!(a["q4"], ("installed".into(), "complete".into()));
    assert_eq!(
        a["q8"],
        ("missing".into(), "incomplete".into()),
        "a torn DiT-only tier must read incomplete, not installed"
    );
    assert_eq!(a["bf16"], ("missing".into(), "missing".into()));
    let disk_bytes = |states: &[Value], tier: &str| {
        states
            .iter()
            .find(|state| state["tier"] == tier)
            .and_then(|state| state["diskSizeBytes"].as_u64())
    };
    assert_eq!(
        disk_bytes(&anima, "q4"),
        Some(3),
        "the complete tier reports its DiT, text encoder, and VAE bytes"
    );
    assert_eq!(
        disk_bytes(&anima, "q8"),
        Some(1),
        "a torn tier reports the bytes that are actually present"
    );
    assert_eq!(disk_bytes(&anima, "bf16"), None);

    // A convert family with no bespoke predicate keeps the coarse backbone probe — a DiT-only tier
    // reads installed (byte-identical to the pre-sc-13513 behavior for any non-anima convert family).
    let coarse = tier_state_map(&mlx_convert_output_tier_states(root, "flux2", "flux2_dev"));
    assert_eq!(coarse["q8"], ("installed".into(), "complete".into()));

    // Mutation check: completing q8 (add the text encoder + VAE) flips it to installed — proving the
    // predicate discriminates on the components, not merely the backbone.
    seed_anima_tier(root, "q8", true);
    let a2 = tier_state_map(&mlx_convert_output_tier_states(root, "anima", "anima_base"));
    assert_eq!(a2["q8"], ("installed".into(), "complete".into()));
}

#[test]
#[cfg(unix)]
fn convert_output_size_counts_shared_directory_symlinks_once() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let tier = tmp.path().join("converted/q4");
    write_weight(&tier, "diffusion_models", "dit.safetensors");
    let shared = tmp.path().join("source/text_encoders");
    std::fs::create_dir_all(&shared).unwrap();
    std::fs::write(shared.join("encoder.safetensors"), b"shared").unwrap();
    symlink(&shared, tier.join("text_encoders")).unwrap();
    // A second alias to the same directory must not double-count the same shared component.
    symlink(&shared, tier.join("text_encoder_alias")).unwrap();

    assert_eq!(
        converted_tier_loaded_bytes(&tier),
        Some(1 + b"shared".len() as u64)
    );
}

// Full catalog path: a converted convert-at-install model (Anima) emits `mlxTiers` from
// `apply_mac_and_mlx_fields`, so /models carries the Studio picker data. macOS-only (the mlx status
// probe is `cfg!(target_os = "macos")`).
#[test]
#[cfg(target_os = "macos")]
fn catalog_emits_mlxtiers_and_states_for_converted_anima() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let converted = data_dir.join("models").join("mlx").join("anima_base");
    std::fs::create_dir_all(&converted).unwrap();
    std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
    // bf16 + q4 fully converted (DiT + text encoder + VAE); q8 is TORN (DiT only) — a realistic
    // interrupted convert. The full `apply_mac_and_mlx_fields` path must carry the family predicate.
    let seed_tier = |tier: &str, complete: bool| {
        let base = converted.join(tier);
        let write = |sub: &str, file: &str| {
            let dir = base.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(file), b"x").unwrap();
        };
        write("diffusion_models", "anima-base-v1.0.safetensors");
        if complete {
            write("text_encoders", "qwen_3_06b_base.safetensors");
            write("vae", "qwen_image_vae.safetensors");
        }
    };
    seed_tier("bf16", true);
    seed_tier("q4", true);
    seed_tier("q8", false);

    let mut object = json!({
        "id": "anima_base",
        "type": "image",
        "family": "anima",
        "mlx": { "requiresConversion": true }
    })
    .as_object()
    .unwrap()
    .clone();
    apply_mac_and_mlx_fields(&mut object, data_dir);
    assert_eq!(
        object.get("mlxConversionState").and_then(Value::as_str),
        Some("converted")
    );
    // `mlxTiers` is the coarse installed-tier list — it drives whether the picker renders; the torn
    // q8 still appears (backbone present) so the user sees it, disabled.
    let tiers: Vec<&str> = object
        .get("mlxTiers")
        .and_then(Value::as_array)
        .expect("mlxTiers emitted")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(tiers, vec!["bf16", "q8", "q4"]);
    // `mlxTierStates` is authoritative: through the full catalog path the family predicate marks the
    // torn q8 incomplete and the complete tiers installed (sc-13513).
    let states = tier_state_map(
        object
            .get("mlxTierStates")
            .and_then(Value::as_array)
            .expect("mlxTierStates emitted"),
    );
    assert_eq!(states["bf16"], ("installed".into(), "complete".into()));
    assert_eq!(states["q4"], ("installed".into(), "complete".into()));
    assert_eq!(states["q8"], ("missing".into(), "incomplete".into()));
    let state_rows = object["mlxTierStates"].as_array().unwrap();
    let disk_bytes = |tier: &str| {
        state_rows
            .iter()
            .find(|state| state["tier"] == tier)
            .and_then(|state| state["diskSizeBytes"].as_u64())
    };
    assert_eq!(disk_bytes("bf16"), Some(3));
    assert_eq!(disk_bytes("q4"), Some(3));
    assert_eq!(disk_bytes("q8"), Some(1));
    // Decoupled from the download matrix — the picker must NOT flip `hasVariantMatrix`.
    assert!(object.get("hasVariantMatrix").is_none());
}

// Per-model quality floor (sc-10731): `apply_mac_and_mlx_fields` surfaces the manifest
// `mlx.minQualityTier` as a top-level `minQualityTier` so the web can clamp the DEFAULT tier up to
// it. Platform-independent (not gated on the macOS mlx-status probe), and only a valid bf16/q8/q4
// value is emitted — a bogus floor is dropped, an absent floor emits nothing.
#[test]
fn catalog_emits_min_quality_floor_from_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let apply = |mlx: Value| {
        let mut object = json!({ "id": "anima_base", "type": "image", "mlx": mlx })
            .as_object()
            .unwrap()
            .clone();
        apply_mac_and_mlx_fields(&mut object, data_dir);
        object
            .get("minQualityTier")
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    // A declared q8 floor is surfaced verbatim as a top-level field.
    assert_eq!(
        apply(json!({ "minQualityTier": "q8" })),
        Some("q8".to_owned())
    );
    // A model with no floor emits nothing (default absent = q4-tolerant, no clamp).
    assert_eq!(apply(json!({ "requiresConversion": true })), None);
    // An invalid floor value is dropped rather than surfaced.
    assert_eq!(apply(json!({ "minQualityTier": "q2" })), None);
}
